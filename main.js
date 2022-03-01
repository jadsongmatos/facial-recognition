let net;

async function app() {
  console.log("Loading mobilenet..");

  // Load the model.
  net = await mobilenet.load({ version: 2, alpha: 1.0 });
  console.log("Successfully loaded model");

  const imgEu = document.getElementById("eu");

  const activation = net.infer(imgEu, true);
  console.log("eu activation", await activation.array());


  // Make a prediction through the model on our image.
  const bill = document.getElementById("bill");
  const inferBill = net.infer(bill, true)
  console.log("bill activation",await inferBill.array() )



  const eu2 = document.getElementById("eu2");
  const inferEu2 = net.infer(eu2, true)
  console.log("eu2 activation", await inferEu2.array())
}

app();
