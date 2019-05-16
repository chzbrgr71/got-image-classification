const fastify = require('fastify')({
    logger: true
})

const gotCharacters = require('./public/data/gotChars.json')
const fetch = require('node-fetch')

//console.log(gotCharacters)

var servingEndpoint = process.env.ML_SERVING_ENDPOINT
//servingEndpoint = "http://gotserving.brianredmond.io:8501/v1/models/inception:predict"
console.log("Tensorflow serving API: " + servingEndpoint)

const fileUpload = require('fastify-file-upload')

const path = require('path')

fastify.register(require('fastify-static'), {
    root: path.join(__dirname, 'public'),
    prefix: '/public/', // optional: default '/'
})

fastify.get('/', function (req, reply) {
    reply.sendFile('index.html') // serving path.join(__dirname, 'public', 'myHtml.html') directly
})

fastify.register(fileUpload)

fastify.post('/upload', function (req, reply) {

    const files = req.raw.files
    var imageFile = files.file.data.toString('base64')
    var postBody = { "inputs": { "image": { "b64": imageFile } } }
    //fetch('http://gotserving.brianredmond.io:8501/v1/models/inception:predict', {
    fetch(servingEndpoint, {        
        method: 'post',
        body: JSON.stringify(postBody),
        headers: { 'Content-Type': 'application/json' },
    })
        .then(res => res.json())
        .then(results => {
            var resArray = results.outputs.prediction[0]
            console.log(resArray);
            let model = resArray.indexOf(Math.max(...resArray))
            console.log(model)
            console.log(resArray[model])
            var pct = resArray[model].toPrecision(3) * 100
            console.log(pct)
            console.log(results.outputs.classes[model])
            reply.send({ payload: gotCharacters[model], sourceImageBase64: imageFile, percentage: pct })
        });
})

fastify.listen(3000, '0.0.0.0', (err, address) => {
    if (err) throw err
    fastify.log.info(`server listening on ${address}`)
})