from importlib import import_module
from zhei.utils.log import Logger
from zhei.utils.catch_error import print_error_info

log = Logger(__name__)

def load_class(class_path: str) -> type:
    """Load a class from a string.

    Args:
        class_path: The class path.

    Returns:
        The class.

    Raises:
        ImportError: If the class cannot be imported.
        AttributeError: If the class cannot be found.
    """
    try:
        module_path, _, class_name = class_path.rpartition(".")
        module = import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        log.error(f"加载类失败: {class_path}")
        print_error_info(e)
        raise e