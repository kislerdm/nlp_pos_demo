# Dmitry Kisler © 2020-present
# www.dkisler.com

import sys
import logging
import inspect


class getLogger:
    def __init__(self,
                 logger: str = "logs",
                 kill: bool = False):
        """Logger class to log and interrupt the process on the error.

        Args: 
          logger: Logger name.
          kill: Flag to terminate the process on the error.
        """
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s.%(msecs)03d [%(levelname)-5s] [%(name)-12s] %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')

        self.logger_name = logger
        self.logs = logging.getLogger(self.logger_name)
        self.kill = kill

    @classmethod
    def get_line(cls):
        """Returns the current line number."""
        return inspect.currentframe().f_back.f_lineno

    def send(self,
             msg: str,
             lineno: int = None,
             kill: bool = None,
             is_error: bool = True) -> None:
        """Send a message into the logging output.

        Args:
          msg: Log message.
          lineno: Line number in the code where log send was triggered.
          kill: Flag to terminate the process on the error.
          is_error: Flag to signal if message is the error message.
        """
        if lineno:
            msg = f"[line: {lineno}] {msg}"
        if is_error:
            self.logs.error(msg)
            if (kill if kill is not None else self.kill):
                sys.exit(1)
        else:
            self.logs.info(msg)
            if kill:
                sys.exit(0)
