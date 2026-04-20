class VideoSummarizerError(Exception):
    """Base exception for the video summarizer project."""
    pass

class VRAMError(VideoSummarizerError):
    """Exception raised for VRAM management errors."""
    pass

class FFmpegError(VideoSummarizerError):
    """Exception raised for FFmpeg operation errors."""
    pass

class IOError(VideoSummarizerError):
    """Exception raised for input/output errors."""
    pass

class ConfigError(VideoSummarizerError):
    """Exception raised for configuration errors."""
    pass

class NoAudioError(VideoSummarizerError):
    """Exception raised when a video file contains no audio track."""
    pass
