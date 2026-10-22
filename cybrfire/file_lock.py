"""Non-blocking advisory batch locks on Windows and POSIX.

The caller owns the file object and keeps it open for the transaction. Closing
it releases the lock. The file must be opened for writing without truncation.
"""
import os

def lock_exclusive(file):
    file.seek(0, os.SEEK_END)
    if file.tell() == 0:
        file.write("0")
        file.flush()
    file.seek(0)
    if os.name == "nt":
        import msvcrt
        try:
            msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise BlockingIOError("Another process holds the batch lock") from exc
    else:
        import fcntl
        fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
