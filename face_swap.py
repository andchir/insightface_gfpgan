import os
import argparse
import cv2
import insightface
from insightface.app import FaceAnalysis

from inference_gfpgan import inference_gfpgan

assert insightface.__version__>='0.7'


def face_swapping(input_path, face_input_path, output_path):
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0, det_size=(640, 640))
    swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=True, download_zip=True)

    img1 = cv2.imread(input_path)
    img2 = cv2.imread(face_input_path)
    faces = app.get(img1)
    faces = sorted(faces, key=lambda x: x.bbox[0])
    source_faces = app.get(img2)
    source_faces = sorted(source_faces, key=lambda x: x.bbox[0])
    print(f'Faces detected: {len(faces)}')
    source_face = source_faces[0]
    bbox = source_face.bbox
    w, h = int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])
    print('Source face size:', w, h)
    res = img1.copy()
    for face in faces:
        res = swapper.get(res, face, source_face, paste_back=True)
    cv2.imwrite(output_path, res)
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--input',
        type=str,
        help='Input image path', required=True)
    parser.add_argument(
        '-fi',
        '--face_input',
        type=str,
        help='Face input image path', required=True)
    parser.add_argument(
        '-o',
        '--output',
        type=str,
        help='Output image path', required=True)
    parser.add_argument(
        '-u',
        '--upscale',
        type=bool,
        default=True,
        help='Upscale faces in the output image')

    args = parser.parse_args()

    output_image_path = face_swapping(args.input, args.face_input, args.output)
    if args.upscale:
        filename, file_extension = os.path.splitext('/path/to/somefile.ext')
        args = {
            'v': '1.4',
            's': 2,
            'input': output_image_path,
            'output': output_image_path
        }
        # output_image_path = inference_gfpgan(args)
    print('Done')
    print('Output: ', output_image_path)


if __name__ == '__main__':
    main()
