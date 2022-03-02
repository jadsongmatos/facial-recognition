var model;
// 1. Setup dataset parameters
const classLabels = ["bill", "eu"];
const imgBill = Array.from(document.querySelectorAll('img[data-train="bill"]'));
const imgEu = Array.from(document.querySelectorAll('img[data-train="eu"]'));

const SEED_WORD = "testSuite";
const LEARNING_RATE = 0.001;
const EPOCHS = 50;

async function app() {
  model = await tmImage.createTeachable(
    { tfjsVersion: tf.version.tfjs },
    { version: 2,alpha:1 }
  );

  model.setLabels(classLabels);
  model.setSeed(SEED_WORD); // set a seed to shuffle predictably

  let time = 0;
  let i = 0

  for (const imgSet of imgBill) {
    let croppedImg = cropTo(imgSet, 224, false);
    await model.addExample(0, croppedImg);
    i++;
    console.log("addExample 0",i);
  }

  for (const imgSet of imgEu) {
    let croppedImg = cropTo(imgSet, 224, false);
    await model.addExample(1, croppedImg);
    i++;
    console.log("addExample 1",i);
  }

  const start = window.performance.now();
  await model.train(
    {
      denseUnits: 100,
      epochs: EPOCHS,
      learningRate: LEARNING_RATE,
      batchSize: 16,
    },
    {
      onEpochBegin: async (epoch,logs) => {
        console.log("Epoch: ", epoch,logs);
      },
      onEpochEnd: async (epoch) => {
        model.stopTraining().then(() => {
          console.log("Stopped training early :", epoch);
        });
      },
    }
  );
  const end = window.performance.now();
  time = end - start;
  console.log(time);

  console.log(await model.predict(document.querySelector("img[data-input]")));

  console.log(await model.predict(document.getElementById("eu3")));
}
app();

const newCanvas = () => document.createElement("canvas");

function cropTo(image, size, flipped = false, canvas = newCanvas()) {
  // image image, bitmap, or canvas
  let width = image.width;
  let height = image.height;

  const min = Math.min(width, height);
  const scale = size / min;
  const scaledW = Math.ceil(width * scale);
  const scaledH = Math.ceil(height * scale);
  const dx = scaledW - size;
  const dy = scaledH - size;
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(image, ~~(dx / 2) * -1, ~~(dy / 2) * -1, scaledW, scaledH);

  // canvas is already sized and cropped to center correctly
  if (flipped) {
    ctx.scale(-1, 1);
    ctx.drawImage(canvas, size * -1, 0);
  }

  return canvas;
}
