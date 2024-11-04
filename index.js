const bodyParser = require("body-parser");
const express = require("express");
const cors = require("cors");
const predict = require("./train");

const app = express();

app.use(
  cors({
    origin: "http://localhost:5173",
  })
);

app.use(express.json());
app.use(bodyParser.json());

app.post("/predict", async (req, res) => {
  const { precipitacion, temperatura, irradiancia, cantidad_paneles } =
    req.body;

  try {
    const result = await predict(
      Number(precipitacion),
      Number(temperatura),
      Number(irradiancia),
      Number(cantidad_paneles)
    );
    console.log("result: ", result);
    return res.status(202).send({ status: "done", result });
  } catch (e) {
    return res.status(500).json({ status: "error", message: e });
  }
});

const PORT = 5000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
