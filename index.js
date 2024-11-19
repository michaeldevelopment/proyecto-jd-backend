const bodyParser = require("body-parser");
const express = require("express");
const cors = require("cors");
const predict = require("./train");
const splitIntoChunks = require("./utils/splitIntoChunks");
const { default: axios } = require("axios");
require("dotenv").config();

const PORT = process.env.PORT || 5000;
const API_KEY = process.env.API_KEY;

const app = express();

app.use(
  cors({
    origin: "*",
    credentials: true,
    optionSuccessStatus: 200,
  })
);

app.use(express.json());
app.use(bodyParser.json());

const isBetween = (x, min, max) => {
  return x >= min && x <= max;
};

app.get("/", (req, res) => {
  res.send("Test API GET server response");
});

app.post("/predict", async (req, res) => {
  try {
    const {
      data: { data },
    } = await axios.get(
      `https://api.weatherbit.io/v2.0/forecast/hourly?lat=7.8891&lon=-72.4967&key=${API_KEY}&hours=${req.body.horas}`
    );

    const primerDia = String(data[0].timestamp_local);
    const tempActual = data[0].app_temp;

    const apiData = data.map((data) => {
      return {
        hora: new Date(data.timestamp_local).getHours(),
        temperatura: data.app_temp,
        precipitacion: data.precip,
      };
    });

    const resultPrediction = await predict(apiData);
    const result = splitIntoChunks(
      apiData.map((element, index) => {
        return {
          ...element,
          irradiancia:
            isBetween(element.hora, 0, 5) || isBetween(element.hora, 19, 23)
              ? 0
              : resultPrediction[index] < 0
              ? 0
              : resultPrediction[index],
        };
      })
    );

    result.shift();
    result.pop();

    console.log("resultado final estimacion => ", result.flat());

    return res.status(202).send({
      status: "done",
      result: result.flat(),
      primerDia,
      tempActual,
    });
  } catch (e) {
    console.log(e);
    return res.status(500).json({ status: "error", message: e });
  }
});

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
