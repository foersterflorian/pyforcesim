import os

from train import (
    BASE_FOLDER,
    FOLDER_MODEL_SAVEPOINTS,
    MODEL_BASE_NAME,
    TIMESTEPS_TOTAL,
    get_save_path_model,
    prepare_model_path,
)


def rename_model() -> None:
    print('Try renaming best model if available...')
    # renaming best model and vec normalize info if present
    model_folder = prepare_model_path(BASE_FOLDER, FOLDER_MODEL_SAVEPOINTS, False)
    save_path_model, save_path_vec_norm = get_save_path_model(
        num_timesteps=(TIMESTEPS_TOTAL + 1), base_name=MODEL_BASE_NAME
    )
    best_model_file = model_folder / 'best_model.zip'
    if best_model_file.exists():
        os.rename(best_model_file, save_path_model)

    best_model_vec_norm_file = model_folder / 'best_model_vec_norm.pkl'
    if best_model_vec_norm_file.exists():
        os.rename(best_model_vec_norm_file, save_path_vec_norm)

    print('Best model renamed successfully')


def main() -> None:
    rename_model()


if __name__ == '__main__':
    main()
