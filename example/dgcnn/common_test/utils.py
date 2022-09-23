


def retrieve_layer_by_name(model:_module_type,layer_name:str):
    """
    Retrieve the layer in the model by the given layer_name

    Args:
        model(Cell):Model which contains the target layer.
        layer_name(str):Name of target layer.

    Returns:
        Cell,the target layer

    Raises:
        ValueError: If module with given layer_name is not found in the model.
    """
    if not isinstance(layer_name,str):
        raise TypeError("la")