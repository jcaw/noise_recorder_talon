from talon import cron, Context, Module, ui, imgui, scope, actions, app, microphone
from talon.lib import flac

# `cubeb` was moved to `lib` on newer Talon
try:
    from talon import cubeb
except ImportError:
    from talon.lib import cubeb
from talon_init import TALON_HOME
from threading import Lock
import re
import time
from pathlib import Path
import os
import subprocess
import inspect
import threading
import random
import uuid
import struct
import logging
from collections import defaultdict


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


# When fullscreen is activated/deactivated, additional changes during the
# deadzone (in seconds) will be ignored.
TRANSITION_DEADZONE = 3
# Recordings shorter than this (in seconds) will not be saved. Quickly exit
# fullscreen to ignore accidental recordings.
MINIMUM_RECORDING_LENGTH = 10
# Long recordings will be split into files of this length.
SPLIT_TIME = "5m"


# Ensure the deadzone won't cause empty recordings to be saved.
assert TRANSITION_DEADZONE <= MINIMUM_RECORDING_LENGTH + 1


IGNORED_MICS = list(
    map(
        re.compile,
        [
            # Add audio sources that you would like to ignore here.
        ],
    )
)


NOISES_ROOT = Path(TALON_HOME, f"recordings/noises/")


def recordings_path(device_name, noise_name):
    """Get the folder a specific noise's recordings should be stored in."""
    # Use only these chars for the mic folder so it works on any file
    # system.
    mic_folder = re.sub("[^a-zA-Z0-9]+", "_", str(device_name))
    MAX_FOLDER_LENGTH = 40
    if len(mic_folder) > MAX_FOLDER_LENGTH:
        mic_folder = mic_folder[:MAX_FOLDER_LENGTH]
    return Path(NOISES_ROOT, mic_folder, str(noise_name))


# FIXME: This seems to be returning wrong values
def get_flac_duration(filename: str) -> float:
    """Returns the duration of a FLAC file in seconds.

    From: https://gist.github.com/lukasklein/8c474782ed66c7115e10904fecbed86a

    """

    def bytes_to_int(bytes: list) -> int:
        result = 0
        for byte in bytes:
            result = (result << 8) + byte
        return result

    with open(filename, "rb") as f:
        if f.read(4) != b"fLaC":
            raise ValueError("File is not a flac file")
        header = f.read(4)
        while len(header):
            meta = struct.unpack("4B", header)  # 4 unsigned chars
            block_type = meta[0] & 0x7F  # 0111 1111
            size = bytes_to_int(header[1:4])

            if block_type == 0:  # Metadata Streaminfo
                streaminfo_header = f.read(size)
                unpacked = struct.unpack("2H3p3p8B16p", streaminfo_header)

                samplerate = bytes_to_int(unpacked[4:7]) >> 4
                sample_bytes = [(unpacked[7] & 0x0F)] + list(unpacked[8:12])
                total_samples = bytes_to_int(sample_bytes)
                duration = float(total_samples) / samplerate

                return duration
            header = f.read(4)


def amount_recorded(device_name, noise_name):
    """Get the amount of `noise_name` on disk for `device`, in seconds."""
    path = recordings_path(device_name, noise_name)
    total_duration = 0.0
    if path.exists():
        for filename in os.listdir(path):
            if filename.endswith(".flac"):
                total_duration += get_flac_duration(path / filename)
    return total_duration


class _RecordingSession(object):
    def __init__(self, device, noise_name, uuid):
        self.device = device
        self.noise_name = noise_name
        self.uuid = uuid
        self._recording = False
        self._lock = Lock()
        self._frames = []
        self._split_cron = None

    def _on_data(self, stream, in_frames, out_frames):
        with self._lock:
            if self._recording:
                self._frames.extend(in_frames)

    def __str__(self):
        return f'<"{self.noise_name}" on "{self.device.name}">'

    def _get_chunk_path(self):
        """Get the path to save the current chunk as."""
        folder = recordings_path(self.device.name, self.noise_name)
        folder.mkdir(parents=True, exist_ok=True)
        # Find first free path number
        index = 0
        while True:
            path = folder / f"{self.uuid}_{index}.flac"
            if path.exists():
                index += 1
            else:
                return path

    def _write_frames(self):
        """Write the frames so far to a file & clear them."""
        # Ignore short recordings, these are probably accidental.
        if len(self._frames) >= 16000 * MINIMUM_RECORDING_LENGTH:
            path = self._get_chunk_path()
            # TODO: Do this on a delay later
            frames = self._frames
            self._frames = []
            # TODO: Spawn thread for this? stopping the stream may also be slow.
            LOGGER.info(f"Writing noise file: {path}")
            flac.write_flac(str(path), frames, compression_level=1)
        else:
            LOGGER.info(
                f"Recording under {MINIMUM_RECORDING_LENGTH} seconds,"
                f" file not written: {self}"
            )

    def finish(self):
        with self._lock:
            LOGGER.info(f"Terminating recording: {self}")
            self._recording = False
            self._write_frames()
            try:
                cron.cancel(self._split_cron)
            except Exception as e:
                LOGGER.info(f"Failed to cancel split cron job: {e}")
            self._split_cron = None
        # This can take a while, so release the lock first
        self._stream.stop()

    def _split_recording(self):
        with self._lock:
            self._write_frames()

    def record(self):
        with self._lock:
            if self._recording:
                raise RuntimeError("Already recording.")

            self._recording = True
            self._frames = []
            ctx = cubeb.Context()
            params = cubeb.StreamParams(
                format=cubeb.SampleFormat.FLOAT32NE,
                rate=16000,
                channels=1,
            )
            existing = amount_recorded(self.device.name, self.noise_name) / 60
            LOGGER.info(
                f"Recording: {self}. {existing:0.1f} mins exist from this device already."
            )
            # TODO: Report how many minutes of this noise have been recorded
            #   already
            self._stream = ctx.new_input_stream(
                f"recording stream - {self.device.name} {self.noise_name}",
                self.device,
                params,
                latency=1,
                data_cb=self._on_data,
            )
            self._stream.start()
            self._split_cron = cron.interval(SPLIT_TIME, self._split_recording)


_active_sessions = []
_sessions_lock = threading.Lock()
# Used by the gui to prompt the user
_gui_text = None
_gui_lock = threading.Lock()


def recording():
    """Is a noise currently being recorded?"""
    with _sessions_lock:
        return bool(_active_sessions)


# TODO: Don't iterate over these, just use a combined regexp
def any_regexp(regexps, string):
    """Are any of `regexps` present in `string`?"""
    for regexp in regexps:
        if regexp.search(string):
            return True
    return False


def _get_free_uuid():
    """Get a noise UUID that hasn't been used by any existing recordings.

    Probability of collision is of course low but this is here just in case.

    """
    existing_paths = _list_recording_paths()
    while True:
        uuid_ = str(uuid.uuid4())
        for path in existing_paths:
            if uuid_ in path:
                continue
        return uuid_


def record(noise_name):
    """Record a noise for `duration` on all input devices."""
    global _active_sessions, _gui_text
    with _sessions_lock:
        if _active_sessions:
            raise RuntimeError("Already recording. End the current recording first.")

        with _gui_lock:
            _gui_text = f'Recording "{noise_name}"...'
        gui.show()

        context = cubeb.Context()
        # HACK: Blunt way to mitigate duplicate devices - Exclude multiple
        #   devices with the same name. Doesn't prevent duplication, just
        #   mitigates it.
        #
        # FIXME: This will exclude actually different devices with the same
        #   name.
        used_names = set()
        # Filename is a UUID, but we use the same UUID for every source. This
        # can be used to cross-reference recordings, e.g. to extract timings
        # from a cleaner mic and apply them to a dirtier mic.
        uuid_ = _get_free_uuid()
        for device in context.inputs():
            if not device.name in used_names and not any_regexp(
                IGNORED_MICS, device.name
            ):
                session = _RecordingSession(device, noise_name, uuid_)
                session.record()
                _active_sessions.append(session)
                used_names.add(device.name)


def stop():
    """End the current recording."""
    global _gui_text, _active_sessions
    with _sessions_lock:
        for session in _active_sessions:
            # Finish can block for a while so spin up a thread to terminate
            # each session.
            thread = threading.Thread(target=session.finish)
            thread.start()
        _active_sessions = []


# Descriptions & previews of each noise can each be found at
# https://noise.talonvoice.com/
#
# Comment out the noises you don't want.
_noises = [
    "clop",
    "fff",
    "ffk",
    "ffp",
    "fft",
    "fuh",
    "hgh",
    "high-fart",
    "hiss",
    "horse",
    "huh",
    "kuh",
    "loogie",
    "low-fart",
    "motorcycle",
    "mouth-smack",
    "oh",
    "pop",
    "pst",
    "puh",
    "rrh",
    "shh",
    "shhk",
    "shhp",
    "smooch",
    "ssk",
    "ssp",
    "sst",
    "sucking-teeth",
    "suh",
    "thh",
    "thhk",
    "thhp",
    "trot",
    "tsk",
    "tss",
    "tuh",
    "uh",
    "xuh",
]


def amounts_recorded_by_device():
    """Get the amount of each noise recorded on each device."""
    devices = {}
    if NOISES_ROOT.exists():
        for device_dir in os.listdir(NOISES_ROOT):
            noises = {}
            for noise_folder in os.listdir(NOISES_ROOT / device_dir):
                noises[noise_folder] = amount_recorded(device_dir, noise_folder)
            devices[device_dir] = noises
    return devices


def amounts_recorded_total():
    """Get the total duration of each noise recorded so far."""
    totals = defaultdict(float)
    for device, noises in amounts_recorded_by_device().items():
        for noise, duration in noises.items():
            totals[noise] += duration
    return totals


def total_data():
    """Return the total amount of all noise recorded, in seconds.

    Includes the same noises recorded across multiple devices. If a device had
    more than one input stream, it may be double counted.

    """
    return sum(amounts_recorded_total().values())


def noise_with_least_data():
    """Get the noise with the lease local data recorded."""
    min_duration = 999999999999999
    min_noise = None
    recorded_amounts = amounts_recorded_total()
    for noise in _noises:
        duration = recorded_amounts.get(noise, 0.0)
        if duration < min_duration:
            min_duration = duration
            min_noise = noise
    return min_noise, min_duration


module = Module()
module.tag(
    "_noise_recorder_context",
    desc="Active when `noise_recorder.py` has a matching context.",
)
module.tag(
    "recording_noises",
    desc=(
        "Active when the noise recorder script is currently recording a noise."
        "\n\nUse to disable Talon acting on noises while recording."
    ),
)


@module.action_class
class NoiseActions:
    def report_noise_recorded() -> None:
        """Pop a notification showing the total amount of noise recorded."""
        mins = total_data() / 60
        hours = mins / 60
        # This double accesses file tree but that's fine
        num_sources = len(amounts_recorded_by_device())
        average_mins = mins / num_sources
        average_hours = hours / num_sources
        report = (
            f"{hours:0.1f} hours total across {num_sources} sources - {average_hours:0.1f}"
            f" hours (or {average_mins:0.0f}) mins per source on average."
        )
        print(f"Total noise recorded: {report}")
        app.notify("Total Noise Recorded", report)



context = Context()
context.matches = r"""
app: /firefox/
app: /chrome/
app: /edge/
app: /safari/
app: /opera/
title: /YouTube/
title: /Vimeo/
title: /Twitch/
"""
# TODO: Disable speech & noises when recording
#
# FIXME: This doesn't capture fullscreen status so you can't just hook behaviour
#   to it.
context.tags = ["user._noise_recorder_context"]


# Used for debouncing
_last_transition = -999
_original_mic = None


def _maybe_record():
    """In the right context, start recording on every mic, otherwise stop."""
    global _last_transition, _original_mic, _gui_text

    # The window dimensions can bounce around during the transitions to & from
    # fullscreen, so deadzones are used for debouncing.
    if (
        "user._noise_recorder_context" in scope.get("tag", [])
        and
        # Assume it's a fullscreen video if the window is on the PRIMARY screen,
        # and matches the fullscreen dimensions. This basically assumes the
        # primary screen has a toolbar.
        ui.active_app().active_window.rect == ui.main_screen().rect
    ):
        if (
            not recording()
            and time.monotonic() > _last_transition + TRANSITION_DEADZONE
        ):
            _last_transition = time.monotonic()
            active_mic = microphone.manager.active_mic()
            _original_mic = active_mic.name if active_mic else None
            print("Disabling mic while recording noises.")
            actions.speech.set_microphone("None")
            with _gui_lock:
                # This can take a while (e.g. on a cold disk drive) so pop a
                # message
                _gui_text = "Scanning noise recordings on disk, this may be slow..."
            gui.show()
            noise, existing = noise_with_least_data()
            LOGGER.info(
                f'Recording noise with the least data: "{noise}", '
                f"{existing / 60:0.1f} mins exist already."
            )
            record(noise)
            context.tags.add("user.recording_noises")
    elif recording() and time.monotonic() > _last_transition + TRANSITION_DEADZONE:
        _last_transition = time.monotonic()
        context.tags.remove("user.recording_noises")
        stop()
        # Lambda is used becayse Python thinks `print_total_noise_recorded`
        # isn't callable.
        cron.after("2s", actions.self.print_total_noise_recorded)
        gui.hide()
        with _gui_lock:
            _gui_text = None
        print("Re-enabling microphone.")
        if _original_mic:
            actions.speech.set_microphone(_original_mic)
            _original_mic = None
        else:
            # Shouldn't ever get here but just use this as a fallback
            print('No previous mic found. Switching to "System Default"')
            actions.speech.set_microphone("System Default")


@imgui.open(software=False, y=0, x=0)
def gui(gui: imgui.GUI):
    global _gui_text
    # TODO: Guard this with a lock?
    with _gui_lock:
        if _gui_text:
            # TODO: Animate this?
            #
            # TODO: Make it red & bold?
            gui.text(_gui_text)


#### Comment out this line to disable the script: ####
cron.interval("100ms", _maybe_record)
