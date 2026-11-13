export function imageData(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(new Error("미리보기 이미지를 읽지 못했습니다."));
    reader.readAsDataURL(blob);
  });
}
