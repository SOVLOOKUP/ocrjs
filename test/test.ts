import fs from 'fs'
import OCR from '../src'
import { base64_to_image } from '@silvia-odwyer/photon-node'

async function main() {
    const ocr = await new OCR().init()
    
    // text direction
    const img1 = readImg(`img/cls1.png`)
    const img2 = readImg(`img/cls2.png`)
    console.log("text cls1:", await ocr.cls(img1.get_base64()))
    console.log("text cls2:", await ocr.cls(img2.get_base64()))
    
    // text det
    // const boarder = readImg(`img/boarder.png`)
    // const points = await ocr.det(img.get_base64())
    // console.log("text det:", points)

    // for (const point of points) {
    //     watermark(img, boarder, point.x ,point.y)
    // }

    // storeImg(img, "out.png")
    
    // text rec
    // console.log("text rec:", await ocr.rec(imgBase64))
    
    // text ocr [todo]
    // console.log("text ocr:", await ocr.ocr(imgBase64))
}

function readImg(path: string) {
    const imgBase64 = fs.readFileSync(path, { encoding: 'base64' })
    const b_data = imgBase64.replace(/^data:image\/(png|jpg);base64,/, "")
    return base64_to_image(b_data)
}

// import { watermark, PhotonImage } from '@silvia-odwyer/photon-node'

// function storeImg(img:PhotonImage,name:string) {
//     const output_base64 = img.get_base64()
//     const output_data = output_base64.replace(/^data:image\/\w+;base64,/, '');
//     fs.writeFile(name, output_data, {encoding: 'base64'}, ()=>{});
// }

main()