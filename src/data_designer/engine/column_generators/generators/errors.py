from data_designer.errors import DataDesignerError


class MissingResourceError(DataDesignerError):
    """Exception for all errors related to missing resources."""


class MissingBlobStorageError(MissingResourceError):
    """Exception for all errors related to missing blob storage."""
