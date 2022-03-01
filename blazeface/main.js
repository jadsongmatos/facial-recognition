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

  const predictionsEu = await model.estimateFaces(
    document.getElementById("eu"),
    returnTensors,
    flipHorizontal,
    annotateBoxes
  );

  console.log(predictionsEu);

  if (predictionsEu.length == 1) {
    if (predictionsEu[0].probability > 90) {
      classifierTop.addExample(predictionsEu[0].topLeft, 0);
      classifierBottom.addExample(predictionsEu[0].bottomRight, 0);
    }
  }

  const predictionsEu2 = await model.estimateFaces(
    document.getElementById("eu2"),
    returnTensors,
    flipHorizontal,
    annotateBoxes
  );

  classifierTop.addExample(predictionsEu2[0].topLeft, 0);
  classifierBottom.addExample(predictionsEu2[0].bottomRight, 0);

  const predictionsBill = await model.estimateFaces(
    document.getElementById("bill"),
    returnTensors,
    flipHorizontal,
    annotateBoxes
  );

  classifierTop.addExample(predictionsBill[0].topLeft, 1);
  classifierBottom.addExample(predictionsBill[0].bottomRight, 1);

  const predictionsBill2 = await model.estimateFaces(
    document.getElementById("bill2"),
    returnTensors,
    flipHorizontal,
    annotateBoxes
  );

  classifierTop.addExample(predictionsBill2[0].topLeft, 1);
  classifierBottom.addExample(predictionsBill2[0].bottomRight, 1);

  const predictionsBill3 = await model.estimateFaces(
    document.getElementById("bill3"),
    returnTensors,
    flipHorizontal,
    annotateBoxes
  );

  classifierTop.addExample(predictionsBill3[0].topLeft, 1);
  classifierBottom.addExample(predictionsBill3[0].bottomRight, 1);

  const predictionsEu3 = await model.estimateFaces(
    document.getElementById("eu3"),
    returnTensors,
    flipHorizontal,
    annotateBoxes
  );

  const resultTop = await classifierTop.predictClass(predictionsEu3[0].topLeft);
  console.log(resultTop);
  const resultBottom = await classifierBottom.predictClass(
    predictionsEu3[0].bottomRight
  );
  console.log(resultBottom);
}

app();
