import tensorflow as tf

save_tflite_path = "./vit_model.tflite"


# Weight Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('./vit_l32.h5')
converter.experimental_new_converter = True
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
with open(save_tflite_path, 'wb') as w:
    w.write(tflite_quant_model)
print("Weight Quantization complete! -", save_tflite_path)