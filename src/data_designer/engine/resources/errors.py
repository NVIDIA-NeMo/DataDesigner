from data_designer.errors import DataDesignerError


class MissingBlobStorageError(DataDesignerError):
    """Exception for all errors related to missing blob storage."""


class MissingManagedAssetsError(DataDesignerError):
    """Exception for all errors related to missing managed assets."""
