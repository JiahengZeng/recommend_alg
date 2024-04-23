import tensorflow as tf

unq_categories = ['music', 'movie', 'finance', 'game', 'military', 'history']
id_mapping_layer = tf.keras.layers.StringLookup(vocabulary=unq_categories)
emb_layer = tf.keras.layers.Embedding(
    input_dim=len(unq_categories) + 1,
    output_dim=4
)

cate_input = tf.constant([['music'], ['finance']])
cate_ids = id_mapping_layer(cate_input)
cate_embeddings = emb_layer(cate_ids)
