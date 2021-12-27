# The error codes.
OK = 0
ALREADY_EXISTS = 1
NOT_FOUND = 2
UNINITIALIZED = 3


class PrunedException(Exception):
    pass


class StatusError(Exception):
    def __init__(self, message: str, error_code: int) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code


class AlreadyExistsError(StatusError):
    def __init__(self, message: str) -> None:
        super().__init__(message, ALREADY_EXISTS)


class NotFoundError(StatusError):
    def __init__(self, message: str) -> None:
        super().__init__(message, NOT_FOUND)


class UnInitializedError(StatusError):
    def __init__(self, message: str) -> None:
        super().__init__(message, UNINITIALIZED)
