[1mdiff --cc main.py[m
[1mindex c1de77f,53bff9c..0000000[m
[1m--- a/main.py[m
[1m+++ b/main.py[m
[36m@@@ -10,23 -8,18 +10,32 @@@[m [mimport numpy as n[m
  import json[m
  import cv2[m
  [m
[32m+ [m
[32m+ [m
  app = Flask(__name__)[m
[32m +load_path = 'skin_model.h5'[m
[32m +global model[m
[32m +model = load_model(load_path, custom_objects={"top_2_accuracy": top_2_accuracy, "top_3_accuracy": top_3_accuracy})[m
[32m +r = "test_image.jpg"[m
  [m
[32m+ [m
  @app.route('/')[m
  def home():[m
      return render_template('index.html')[m
  [m
[32m++<<<<<<< HEAD[m
[32m +@app.route('/predict/',methods=['POST'])[m
[32m +def predict():[m
[32m +    # data = request.get_json() ###################################[m
[32m +    # print(data) ################################################[m
[32m +    pass[m
[32m +[m
[32m +def predict_one(img, model=model, print_all=False, plot_img=False):[m
[32m++=======[m
[32m+ [m
[32m+ @app.route('/predict/', methods=['POST'])[m
[32m+ def predict_one(img, model, print_all=False, plot_img=False):[m
[32m++>>>>>>> d4098321d9a453252ec476a544c3a49215261d50[m
      resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)[m
      preprocessed = preprocess_input(resized)[m
      input_img = preprocessed.reshape(1, 224, 224, 3)[m
[36m@@@ -65,8 -58,9 +74,9 @@@[m
          plt.title('Mole')[m
          plt.show()[m
  [m
[31m -    return (pred_name_class, pred_class, pred_R)[m
[32m +    return jsonify((pred_name_class, pred_class, pred_R))[m
  [m
[32m+ [m
  # returns a compiled model[m
  def top_3_accuracy(y_true, y_pred):[m
      return top_k_categorical_accuracy(y_true, y_pred, k=3)[m
[36m@@@ -74,32 -69,10 +85,39 @@@[m
  def top_2_accuracy(y_true, y_pred):[m
      return top_k_categorical_accuracy(y_true, y_pred, k=2)[m
  [m
[32m++<<<<<<< HEAD[m
[32m +def url2rgb(url, background=(255,255,255) ):[m
[32m +    """Image converting in case if we get a link"""[m
[32m +    image_np = io.imread(url)[m
[32m +    row, col, ch = image_np.shape[m
[32m +[m
[32m +    if ch == 3:[m
[32m +        return rgba[m
[32m +[m
[32m +    assert ch == 4, 'RGBA image has 4 channels.'[m
[32m +[m
[32m +    rgb = np.zeros( (row, col, 3), dtype='float32' )[m
[32m +    r, g, b, a = rgba[:,:,0], rgba[:,:,1], rgba[:,:,2], rgba[:,:,3][m
[32m +[m
[32m +    a = np.asarray( a, dtype='float32' ) / 255.0[m
[32m +[m
[32m +    R, G, B = background[m
[32m +[m
[32m +    rgb[:,:,0] = r * a + (1.0 - a) * R[m
[32m +    rgb[:,:,1] = g * a + (1.0 - a) * G[m
[32m +    rgb[:,:,2] = b * a + (1.0 - a) * B[m
[32m +[m
[32m +    return np.asarray(rgb, dtype='uint8')[m
[32m++=======[m
[32m++>>>>>>> d4098321d9a453252ec476a544c3a49215261d50[m
  [m
  if __name__ == '__main__':[m
      print('Main')[m
[31m -    load_path = 'skin_model.h5'[m
[31m -    model = load_model(load_path, custom_objects={"top_2_accuracy": top_2_accuracy, "top_3_accuracy": top_3_accuracy})[m
      print('Model is loaded', type(model))[m
[31m -    app.run(host='0.0.0.0')[m
[32m++<<<<<<< HEAD[m
[32m +    app.run(debug=True, host='0.0.0.0')[m
[32m +[m
[32m +[m
[32m++=======[m
[32m++    app.run(host='0.0.0.0')[m
[32m++>>>>>>> d4098321d9a453252ec476a544c3a49215261d50[m
