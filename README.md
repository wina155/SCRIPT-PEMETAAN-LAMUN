// ===========================
// Step 1: Area dan Tanggal
// ===========================
var area = AREA; // Ganti dengan geometry area of interest
var startDate = '2024-01-01';
var endDate = '2024-12-30';
var pixelArea = ee.Image.pixelArea();
Map.centerObject(area, 15);

// ===========================
// Step 2: Masking Awan
// ===========================
function maskS2clouds(image) {
  var cloudProb = image.select('MSK_CLDPRB');
  var mask = cloudProb.lt(20);
  return image.updateMask(mask)
              .divide(10000)
              .select('B.*')
              .copyProperties(image, ['system:time_start']);
}

// ===========================
// Step 3: Load Citra Sentinel-2
// ===========================
var sr2ACol = ee.ImageCollection('COPERNICUS/S2_SR')
  .filterBounds(area)
  .filterDate(startDate, endDate)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
  .map(maskS2clouds)
  .median()
  .clip(area);

Map.addLayer(sr2ACol, {bands: ['B4', 'B3', 'B2'], min: 0, max: 0.3}, 'Citra RGB');

// ===========================
// Step 4: NDWI
// ===========================
var ndwi = sr2ACol.select('B3').subtract(sr2ACol.select('B8'))
  .divide(sr2ACol.select('B3').add(sr2ACol.select('B8')))
  .rename('NDWI');

var ndwiSmooth = ndwi.focal_mean(1, 'square', 'pixels');
Map.addLayer(ndwi, {min: -1, max: 1, palette: ['brown', 'white', 'blue']}, 'NDWI');

// ===========================
// Step 5: Klasifikasi NDWI
// ===========================
var thresholdShallow = 0.12;
var thresholdWater = 0.00;

var land = ndwiSmooth.lte(thresholdWater);
var shallowWater = ndwiSmooth.gt(thresholdShallow);
var deepWater = ndwiSmooth.gt(thresholdWater).and(ndwiSmooth.lte(thresholdShallow));

// ===========================
// Step 6: Perluasan Laut Dangkal (20 meter)
// ===========================
var landBuffered = land.focal_max(2);  // Buffer darat (2 piksel ~ 20 m)
var shallowWaterExpanded = shallowWater.focal_max({ radius: 20, units: 'meters' });
var shallowWaterFinal = shallowWaterExpanded.and(landBuffered.not());
var deepWaterFinal = deepWater.and(landBuffered.not());

Map.addLayer(landBuffered.updateMask(landBuffered), {palette: ['black']}, 'Darat');
Map.addLayer(shallowWaterFinal.updateMask(shallowWaterFinal), {palette: ['#006400']}, 'Perairan Dangkal (Expanded)');
Map.addLayer(deepWaterFinal.updateMask(deepWaterFinal), {palette: ['#87CEFA']}, 'Perairan Dalam');

// ===========================
// Step 7: Gabungan Klasifikasi NDWI
// ===========================
var klasifikasiGabungan = landBuffered.multiply(1)
  .where(shallowWaterFinal, 2)
  .where(deepWaterFinal, 3)
  .rename('Klasifikasi_NDWI');

Map.addLayer(klasifikasiGabungan, {
  min: 1,
  max: 3,
  palette: ['black', '#006400', '#87CEFA']
}, 'Klasifikasi NDWI');

// ===========================
// Step 8: Smoothing
// ===========================
var klasifikasiSmooth = klasifikasiGabungan.focal_mode({
  radius: 2,
  units: 'pixels',
  iterations: 1
}).rename('Klasifikasi_Smooth');

Map.addLayer(klasifikasiSmooth, {
  min: 1,
  max: 3,
  palette: ['black', '#006400', '#87CEFA']
}, 'Klasifikasi NDWI Smooth');

// ===========================
// Step 9: Statistik NDWI
// ===========================
var ndwiStats = ndwi.reduceRegion({
  reducer: ee.Reducer.minMax().combine({reducer2: ee.Reducer.mean(), sharedInputs: true}),
  geometry: area,
  scale: 10,
  maxPixels: 1e9
});
print('Statistik NDWI:', ndwiStats);

// ===========================
// Step 10: Koreksi Permukaan
// ===========================
var image = sr2ACol.select(["B1", "B2", "B3", "B4", "B8"]).clip(area);
var X1cor = image.select('B1').subtract(0.18450).log().rename('X1');
var X2cor = image.select('B2').subtract(0.17634).log().rename('X2');
var X3cor = image.select('B3').subtract(0.17381).log().rename('X3');
var X4cor = image.select('B4').subtract(0.15420).log().rename('X4');
var Xij = X1cor.addBands(X2cor).addBands(X3cor).addBands(X4cor);

// ===========================
// Step 11: Masking dan Pelatihan RF Darat
// ===========================
var DATALAPANGANNEW = ee.FeatureCollection('projects/ee-winaapriani62377/assets/DATALAPANGANNEW');
var XijDarat = Xij.updateMask(landBuffered);

var trainingData = XijDarat.sampleRegions({
  collection: DATALAPANGANNEW,
  properties: ['class'],
  scale: 10
}).filter(ee.Filter.notNull(['X1', 'X2', 'X3', 'X4']));

var bands = ['X1', 'X2', 'X3', 'X4'];
var classifierRF = ee.Classifier.smileRandomForest(50).train({
  features: trainingData,
  classProperty: 'class',
  inputProperties: bands
});

var classifiedImageRF = XijDarat.select(bands).classify(classifierRF);
Map.addLayer(classifiedImageRF, {
  min: 0,
  max: 3,
  palette: ['#006400', '#EEDC82', '#654321', 'gray']
}, 'Hasil Klasifikasi RF (Darat)');

// ===========================
// Step 12: Evaluasi Akurasi Darat
// ===========================
var sampledData = trainingData.randomColumn();
var trainingSet = sampledData.filter(ee.Filter.lt('random', 0.7));
var testSet = sampledData.filter(ee.Filter.gte('random', 0.7));
var testClassification = testSet.classify(classifierRF);
var confusionMatrixRF = testClassification.errorMatrix('class', 'classification');
print('Confusion Matrix RF:', confusionMatrixRF);
print('Overall Accuracy RF:', confusionMatrixRF.accuracy());
print('Kappa RF:', confusionMatrixRF.kappa());

// ===========================
// Step 13: RF untuk Laut Dangkal
// ===========================
var XijLaut = Xij.updateMask(shallowWaterFinal);
var trainingLaut = XijLaut.sampleRegions({
  collection: DATALAPANGANNEW,
  properties: ['class'],
  scale: 10
}).filter(ee.Filter.notNull(['X1', 'X2', 'X3', 'X4']));

var classifierRFLaut = ee.Classifier.smileRandomForest(50).train({
  features: trainingLaut,
  classProperty: 'class',
  inputProperties: bands
});

var classifiedImageRFLaut = XijLaut.select(bands).classify(classifierRFLaut);
Map.addLayer(classifiedImageRFLaut, {
  min: 0,
  max: 3,
  palette: ['#006400', '#EEDC82', '#654321', 'gray']
}, 'Hasil Klasifikasi RF (Laut Dangkal)');

// ===========================
// Step 14: Evaluasi Akurasi Laut
// ===========================
var sampledLaut = trainingLaut.randomColumn();
var trainLaut = sampledLaut.filter(ee.Filter.lt('random', 0.7));
var testLaut = sampledLaut.filter(ee.Filter.gte('random', 0.7));
var testClassificationLaut = testLaut.classify(classifierRFLaut);
var confusionMatrixLaut = testClassificationLaut.errorMatrix('class', 'classification');
print('Confusion Matrix RF (Laut):', confusionMatrixLaut);
print('Overall Accuracy RF (Laut):', confusionMatrixLaut.accuracy());
print('Kappa RF (Laut):', confusionMatrixLaut.kappa());

// ===========================
// Step 15: Hitung Luas Tiap Kelas
// ===========================
function calculateAreaPerClass(img, classValue) {
  var mask = img.eq(classValue);
  var stats = pixelArea.updateMask(mask).reduceRegion({
    reducer: ee.Reducer.sum(),
    geometry: area,
    scale: 10,
    maxPixels: 1e13
  });
  return ee.Number(stats.get('area')).divide(10000);
}

[0, 1, 2, 3].forEach(function(classValue) {
  print('Luas kelas darat ' + classValue + ' (ha):', calculateAreaPerClass(classifiedImageRF, classValue));
  print('Luas kelas laut ' + classValue + ' (ha):', calculateAreaPerClass(classifiedImageRFLaut, classValue));
});

// ===========================
// Step 16: Legend
// ===========================
function makeRow(color, name) {
  var colorBox = ui.Label({
    style: {
      backgroundColor: color,
      padding: '8px',
      margin: '2px 6px 2px 0',
      border: '1px solid #000'
    }
  });
  var description = ui.Label({
    value: name,
    style: { margin: '4px 0' }
  });
  return ui.Panel({
    widgets: [colorBox, description],
    layout: ui.Panel.Layout.Flow('horizontal')
  });
}

var legend = ui.Panel({
  style: {
    position: 'bottom-left',
    padding: '8px 15px',
    backgroundColor: 'rgba(255,255,255,0.8)'
  }
});

legend.add(ui.Label({
  value: 'Legenda Klasifikasi',
  style: { fontWeight: 'bold', fontSize: '16px', margin: '0 0 6px 0' }
}));

legend.add(makeRow('black', 'Darat'));
legend.add(makeRow('#006400', 'Perairan Dangkal'));
legend.add(makeRow('#87CEFA', 'Perairan Dalam'));
legend.add(ui.Label('--- Klasifikasi Habitat Laut ---', {margin: '6px 0 4px 0'}));
legend.add(makeRow('#006400', 'Lamun'));
legend.add(makeRow('#EEDC82', 'Pasir'));
legend.add(makeRow('#654321', 'Lumpur'));
legend.add(makeRow('gray', 'Batu'));

Map.add(legend);

// ===========================
// Step 17: Export
// ===========================
Export.image.toDrive({
  image: classifiedImageRF,
  description: 'pemetaan_2024',
  scale: 10,
  region: AREA,
  fileFormat: 'GeoTIFF'
});
