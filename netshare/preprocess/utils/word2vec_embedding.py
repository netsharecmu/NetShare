import os

from gensim.models import Word2Vec

from netshare.utils.logger import logger


def word2vec_train(
    df,
    out_dir,
    model_name,
    word2vec_cols,
    word2vec_size,
    annoy_n_trees,
    force_retrain=False,  # retrain from scratch
    model_test=False,
):
    model_path = os.path.join(out_dir, "{}_{}.model".format(model_name, word2vec_size))

    if os.path.exists(model_path) and not force_retrain:
        logger.info("Loading Word2Vec pre-trained model...")
        model = Word2Vec.load(model_path)
    else:
        logger.info("Training Word2Vec model from scratch...")
        sentences = []
        for row in range(0, len(df)):
            sentence = [
                str(df.at[row, col]) for col in [c.column for c in word2vec_cols]
            ]
            sentences.append(sentence)

        model = Word2Vec(
            sentences=sentences, size=word2vec_size, window=5, min_count=1, workers=10
        )
        model.save(model_path)
    logger.info(f"Word2Vec model is saved at {model_path}")

    return model_path
