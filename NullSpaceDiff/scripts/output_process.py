import cv2
import os
import glob


def main(input_dir, output_dir, target_width, target_height):
    os.makedirs(output_dir, exist_ok=True)

    extensions = ['*.jpg', '*.jpeg', '*.png']

    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))

    print(f"対象ディレクトリ: {input_dir}")
    print(f"検出された画像数: {len(image_paths)} 枚")
    print(f"目標サイズ: {target_width}x{target_height}")
    print("-" * 30)

    for img_path in image_paths:
        # 画像の読み込み
        img = cv2.imread(img_path)

        if img is None:
            print(f"[Skip] 読み込みに失敗しました: {img_path}")
            continue

        # リサイズ処理 (cv2.resizeは (width, height) の順で指定します)
        # interpolationは縮小ならcv2.INTER_AREA、拡大ならcv2.INTER_LINEARなどが一般的です
        resized_img = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_AREA)
        resized_img = resized_img[60:270, 62:442]

        # 保存先パスの作成
        filename = os.path.basename(img_path)
        save_path = os.path.join(output_dir, filename)

        # 画像の保存
        cv2.imwrite(save_path, resized_img)
        print(f"[OK] 保存完了: {save_path}")

    print("-" * 30)
    print("すべての処理が完了しました。")



if __name__ == "__main__":
    INPUT_DIR = f"./output"
    OUTPUT_DIR = f"./output_resized"

    WIDTH = 480
    HEIGHT = 270

    main(INPUT_DIR, OUTPUT_DIR, WIDTH, HEIGHT)