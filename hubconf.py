dependencies = ['torch','torchaudio']

from  ptcrepe.crepe import CREPE


def load_crepe(model_capacity="full"):
    """
    Inits and return a Crepe instance.

    Args:
        model_capacity (str): type of model, can be 'full', 'large', 'medium', 'small', 'tiny'
    Returns:
        object: Crepe model object
    """

    return CREPE(model_capacity=model_capacity)