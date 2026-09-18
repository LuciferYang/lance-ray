import pytest
from lance_ray.utils import has_namespace_params, validate_uri_or_namespace


def test_has_namespace_params_treats_empty_values_as_not_provided() -> None:
    assert has_namespace_params("", ["table"]) is False
    assert has_namespace_params("dir", []) is False
    assert has_namespace_params("dir", ["table"]) is True
    assert has_namespace_params(None, None) is False


def test_validate_uri_or_namespace_rejects_empty_namespace_params() -> None:
    # An empty namespace_impl or table_id (e.g. from os.environ.get) must be
    # rejected at the API boundary, not deep inside lance_namespace.connect().
    with pytest.raises(ValueError, match="Must provide either 'uri' OR"):
        validate_uri_or_namespace(None, "", [])
