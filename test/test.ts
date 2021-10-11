import fs from 'fs'
import OCR from '../src'

async function main() {
    const ocr = await new OCR().init()
    const imgBase64 = fs.readFileSync(`img/test.png`, { encoding: 'base64' })
    
    // text direction
    // console.log("text cls:", await ocr.cls(imgBase64))
    
    // text det
    console.log("text det:", await ocr.det(imgBase64))

    // text rec
    // console.log("text rec:", await ocr.rec(imgBase64))
    
    // text ocr [todo]
    // console.log("text ocr:", await ocr.ocr(imgBase64))
}

main()