"""
Sets the devices for training scripts.
"""

import tensorflow as tf


def set_gpu(gpu_ids_list):
    """
    Set the visible devices from a list of GPUs. Used to assign a process to a separate GPU.
    
    Also very important is to restrict memeory growth since a single tensorfow process will allocate almost all 
    GPU memory, so two fits can not run on same GPU.

    Args:
        gpu_ids_list (list): Device list.

    Returns:
        None.

    """
    # Check if set is possible
    if len(gpu_ids_list) <= 0:
        print("Info: No gpu to set")
        return

    if tf.test.is_built_with_gpu_support() is False and tf.test.is_built_with_cuda() is False:
        print("Warning: No cuda support")
        print("Warning: Can not set GPU")
        return

    try:
        gpus = tf.config.list_physical_devices('GPU')
    except:
        print("Error: Can not get device list, do nothing")
        return

    if isinstance(gpus, list):
        if len(gpus) <= 0:
            print("Warning: No devices found")
            print("Warning: Can not set GPU")
            return
        try:
            gpus_used = [gpus[i] for i in gpu_ids_list if 0 <= i < len(gpus)]
            tf.config.set_visible_devices(gpus_used, 'GPU')
            print("Info: Setting visible devices: ", gpus_used)
            for gpu in gpus_used:
                print("Restrict Memory:", gpu)
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print("Info:", len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)
