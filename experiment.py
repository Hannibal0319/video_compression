from eval import evaluate_video
from video_encoding import encode_video

def run_experiment(input_video, output_video, codec, method):
    # Step 1: Encode the video using the specified method
    encode_video(input_video, output_video, codec=codec, method=method)

    # Step 2: Evaluate the encoded video against the original
    avg_psnr, avg_ssim = evaluate_video(input_video, output_video)

    return avg_psnr, avg_ssim

if __name__ == "__main__":
    input_video = './videos/Jockey_1920x1080_120fps_420_8bit_YUV.yuv'
    output_video = f'./compressed_videos/{input_video.split("/")[-1].split(".")[0]}_compressed.avi'
    codec = 'XVID'
    method = 'basic'  # Options: 'basic', 'grayscale', 'blur', 'edge_detection', 'resize_half'

    psnr, ssim = run_experiment(input_video, output_video, codec, method)
    print(f"Average PSNR: {psnr:.2f} dB")
    print(f"Average SSIM: {ssim:.4f}")