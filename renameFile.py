import os


def renameFile(path):
    for file in os.listdir(path):
        filename = os.path.splitext(file)[0]

        filename = filename[11:]

        ann = str(filename) + ".txt"

        os.rename(
            os.path.join(path, file),
            os.path.join(path, ann)
        )


if __name__ == '__main__':
    path_ann = None
    renameFile(path_ann)