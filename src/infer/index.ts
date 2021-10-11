import { Tensor } from "onnxruntime-common"
import { InferInput } from "../init"
import { infer, yResize } from "./utils"

// 文本方向分类
async function clsInfer(opt: InferInput) {
  // [x,2]
  const postInfer = (
    dataC: Tensor,
  ) => (dataC.data[0] >= dataC.data[1]
    ? { "direction": 0, "confidence": dataC.data[0] }
    : { "direction": 180, "confidence": dataC.data[1] })

  return await infer(opt, postInfer)
}

// 文本检测
async function detInfer(opt: InferInput, resizeMethod: 1 | 2 | 3 | 4 | 5) {
  const { photon } = opt
  const { resize } = photon

  // wh 必须为 32 倍数
  const preInfer = (img: import("@silvia-odwyer/photon").PhotonImage) =>
    resize(img, yResize(img.get_width()), yResize(img.get_height()), resizeMethod)

  // [x,1,w,h] todo 解析结果
  const postInfer = (dataC: Tensor) => JSON.stringify(dataC)

  return await infer(opt, postInfer, preInfer)
}

// 文本识别
async function recInfer(opt: InferInput, resizeMethod: 1 | 2 | 3 | 4 | 5) {
  const { photon } = opt
  const { resize } = photon

  // wh 必须为 32
  const preInfer = (img: import("@silvia-odwyer/photon").PhotonImage) =>
    resize(img, 32, 32, resizeMethod)

  // [x,8,6624] todo 解析结果
  const postInfer = (dataC: Tensor) => JSON.stringify(dataC)

  return await infer(opt, postInfer, preInfer)
}

export { clsInfer, detInfer, recInfer }
