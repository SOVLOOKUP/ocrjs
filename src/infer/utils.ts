import { InferInput } from "../init"
import { Feeds, PostInfer, PreInfer } from "./types"

export async function infer<T>(opt: InferInput, postInfer: PostInfer<T>, preInfer?: PreInfer) {
    const { base64, modelSession, onnxruntime, photon } = opt
    const { Tensor } = onnxruntime
    const { PhotonImage } = photon
    const data = base64.replace(/^data:image\/(png|jpg);base64,/, "")
    let phtn_img = PhotonImage.new_from_base64(data)
    if (preInfer !== undefined) phtn_img = await preInfer(phtn_img)
    const width = phtn_img.get_width()
    const height = phtn_img.get_height()
    let img = phtn_img.get_raw_pixels()
    img = img.filter((_: number, i: number) => {
        // 索引 +1 不是 4 的倍数 去掉 alpha 通道
        return ((i + 1) % 4 !== 0)
    })
    img = img.map((v) => {
        // 归一化
        return v / 255
    })
    try {
        const dims = [1, 3, width, height]
        const inputData = Float32Array.from(img)
        // 输入模型的数据
        const feeds = <Feeds>{}
        feeds[modelSession.inputNames[0]] = new Tensor('float32', inputData, dims)
        // 进行模型推理
        const results = await modelSession.run(feeds)
        // 读取结果
        const dataC = results[modelSession.outputNames[0]]
        return await postInfer(dataC)
    } catch (e) {
        throw e
    }
}

export function yResize(n: number) {
    n = n < 32 ? 32 : n
    const y = n % 32
    if ((32 - (2 * y)) > 0) {
        n = n - y
    } else {
        n = n + 32 - y
    }
    return n
}