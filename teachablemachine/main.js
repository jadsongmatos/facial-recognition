var model;
// 1. Setup dataset parameters
const classLabels = ["angelina_jolie", "bill_gates", "neymar"];

const SEED_WORD = "testSuite";
const LEARNING_RATE = 0.001;
const EPOCHS = 100;

const surface = { name: "show.fitCallbacks", tab: "Training" };

const callbacks = {
  onEpochEnd: async (epoch, logs) => {
    tfvis.show.fitCallbacks(epoch, logs);
    console.log("epoch: " + epoch + JSON.stringify(logs));
  },
};

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
    [224, 224]
  );
  return crop.div(255);
}

async function app() {
  model = await tmImage.createTeachable(
    { tfjsVersion: tf.version.tfjs },
    { version: 2, alpha: 1 }
  );

  model.setLabels(classLabels);

  //tfvis.show.modelSummary(surface, model);

  model.setSeed(SEED_WORD); // set a seed to shuffle predictably

  await model.addExample(
    0,
    preprocess(tf.browser.fromPixels(document.getElementById("angelina_jolie")))
  );
  await model.addExample(
    1,
    preprocess(tf.browser.fromPixels(document.getElementById("bill_gates")))
  );
  await model.addExample(
    2,
    preprocess(tf.browser.fromPixels(document.getElementById("neymar")))
  );

  const start = window.performance.now();
  await model.train(
    {
      denseUnits: 100,
      epochs: EPOCHS,
      learningRate: LEARNING_RATE,
      batchSize: 16,
      //callbacks: tfvis.show.fitCallbacks(surface, ["loss", "acc"]), //callbacks
    },
    {
      onEpochBegin: async (epoch, logs) => {
        console.log("Epoch: ", epoch);
      },
    }
  );

  console.log(window.performance.now() - start);

  const model_predict_input = await model.predict(
    document.getElementById("lenna")
  );

  console.log(model_predict_input);

  let result_input = 0;
  let max = 1;
  model_predict_input.forEach((categories) => {
    result_input = result_input + categories.probability;
    if (categories.probability > max) {
      max = categories.probability;
    }
  });

  console.log("result:", result_input / max, "max:", max);
}
app();
