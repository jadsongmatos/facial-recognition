var model;
const classifier = knnClassifier.create();

function preprocess(imageTensor) {
  const widthToHeight = imageTensor.shape[1] / imageTensor.shape[0];
  let squareCrop;
  if (widthToHeight > 1) {
    const heightToWidth = imageTensor.shape[0] / imageTensor.shape[1];
    const cropTop = (1 - heightToWidth) / 2;
    const cropBottom = 1 - cropTop;
    squareCrop = [[cropTop, 0, cropBottom, 1]];
  } else {
    const cropLeft = (1 - widthToHeight) / 2;
    const cropRight = 1 - cropLeft;
    squareCrop = [[0, cropLeft, 1, cropRight]];
  }
  // Expand image input dimensions to add a batch dimension of size 1.
  const crop = tf.image.cropAndResize(
    tf.expandDims(imageTensor),
    squareCrop,
    [0],
    [480, 480]
  );
  return crop.div(255);
}

async function app() {
  // Load the model.
  model = await tf.loadGraphModel(
    "/efficientnet_v2_imagenet21k_l/model/model.json"
  );

  console.log("Successfully loaded model");

  let start = window.performance.now();
  let end;

  classifier.addExample(model.predict(
    preprocess(tf.browser.fromPixels(document.getElementById("angelina_jolie")))
  ), "angelina_jolie");

  end = window.performance.now();
  console.log("angelina_jolie", end - start);
  start = window.performance.now();

 
  classifier.addExample(model.predict(
    preprocess(tf.browser.fromPixels(document.getElementById("bill_gates")))
  ), "bill_gates");

  end = window.performance.now();
  console.log("bill_gates", end - start);
  start = window.performance.now();

  classifier.addExample(model.predict(
    preprocess(tf.browser.fromPixels(document.getElementById("neymar")))
  ), "neymar");

  end = window.performance.now();
  console.log("neymar", end - start);
  start = window.performance.now();

  const input = model.predict(
    preprocess(tf.browser.fromPixels(document.getElementById("lenna")))
  );
  console.log(input);

  const model_predict_input = await classifier.predictClass(input,16);
  console.log(model_predict_input);

  let result_input = 0;
  let max = 0;
  Object.values(model_predict_input.confidences).forEach((categories) => {
    result_input = result_input + categories;
    if (categories > max) {
      max = categories;
    }
  });

  console.log("result:", result_input / max, "max:", max);
}

app();
