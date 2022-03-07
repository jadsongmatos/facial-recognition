var model;
const classifierTop = knnClassifier.create();
const classifierBottom = knnClassifier.create();

async function app() {
  // Load the model.
  model = await blazeface.load();
  console.log("Successfully loaded model");

  const returnTensors = true;
  const flipHorizontal = false;
  const annotateBoxes = true;

  let predictions

  predictions = await model.estimateFaces(
    document.getElementById("angelina_jolie"),
    returnTensors,
    flipHorizontal,
    annotateBoxes
  );

  if (predictions.length == 1) {
    if (predictions[0].probability > 90) {
      classifierTop.addExample(predictions[0].topLeft, "angelina_jolie");
      classifierBottom.addExample(predictions[0].bottomRight, "angelina_jolie");
    }
  }

  predictions = await model.estimateFaces(
    document.getElementById("bill_gates"),
    returnTensors,
    flipHorizontal,
    annotateBoxes
  );

  classifierTop.addExample(predictions[0].topLeft, "bill_gates");
  classifierBottom.addExample(predictions[0].bottomRight, "bill_gates");

  predictions = await model.estimateFaces(
    document.getElementById("neymar"),
    returnTensors,
    flipHorizontal,
    annotateBoxes
  );

  classifierTop.addExample(predictions[0].topLeft, "neymar");
  classifierBottom.addExample(predictions[0].bottomRight, "neymar");

  predictions = await model.estimateFaces(
    document.getElementById("lenna"),
    returnTensors,
    flipHorizontal,
    annotateBoxes
  );

  const resultTop = await classifierTop.predictClass(predictions[0].topLeft);
  console.log(resultTop);
  //console.log("top activation", await predictions[0].topLeft.array());

  const resultBottom = await classifierBottom.predictClass(
    predictions[0].bottomRight
  );
  console.log(resultBottom);
  //console.log("bottom activation", await predictions[0].bottomRight.array());
}

app();
