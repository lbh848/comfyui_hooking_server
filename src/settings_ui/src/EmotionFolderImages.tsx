import { useEffect, useRef, useState } from "react";
import { RefreshCw, Trash2, Upload } from "lucide-react";
import { Empty, Modal } from "./components";
import { IllustrationFeedback, IllustrationLoading } from "./IllustBackups";
import { query, useIllustrationAction, type IllustrationData } from "./illustration";
import { useBotApi, useBotResource } from "./BotApiScope";
import "./asset.css";

type Props = {
  botName: string; character: string; profile: string; outfit: string; emotion: string;
  readOnly?: boolean; allowUpload?: boolean; onChanged?: () => void;
};

/** 한 감정 폴더(또는 미분류 캐릭터 루트)의 이미지를 보여주고 업로드·삭제를 처리한다. */
export function EmotionFolderImages({ botName, character, profile, outfit, emotion, readOnly, allowUpload = true, onChanged }: Props) {
  const api = useBotApi();
  const assets = useBotResource(`bot_mode/emotion_assets?${query({ bot: botName, character, profile, outfit, emotion })}`, true);
  const action = useIllustrationAction();
  const images: IllustrationData[] = assets.data?.images || [];
  const [preview, setPreview] = useState<IllustrationData | null>(null);
  const [deletingImage, setDeletingImage] = useState<IllustrationData | null>(null);
  const uploadInput = useRef<HTMLInputElement>(null);
  const location = profile ? `${profile} · ${outfit} · ${emotion}` : "전체 이미지";
  useEffect(() => {
    // 생성 작업이 백업으로 저장되면 열려 있는 이미지 그리드도 즉시 다시 읽는다.
    const reload = () => assets.reload();
    window.addEventListener("emotion-store-changed", reload);
    return () => window.removeEventListener("emotion-store-changed", reload);
  }, [assets.reload]);

  async function upload(files: FileList | null) {
    if (!files?.length || readOnly || !allowUpload) return;
    const formFiles = Array.from(files);
    await action.run(async () => {
      for (const file of formFiles) {
        const form = new FormData();
        form.set("bot", botName);
        form.set("character", character);
        form.set("profile", profile);
        form.set("outfit", outfit);
        form.set("emotion", emotion);
        form.set("file", file);
        await api.request("bot_mode/emotion_upload", form);
      }
      assets.reload(); onChanged?.();
    }, `${formFiles.length}장을 업로드했습니다.`);
    if (uploadInput.current) uploadInput.current.value = "";
  }

  async function removeImage(image: IllustrationData) {
    if (readOnly) return;
    await action.run(async () => {
      await api.request("bot_mode/emotion_assets", {
        action: "remove", bot: botName, character, profile, outfit, emotion, filename: image.filename,
      });
      setDeletingImage(null); setPreview(null);
      assets.reload(); onChanged?.();
    }, "이미지를 삭제했습니다.");
  }

  return <div className="illust-form">
    <div className="illust-actions">
      <span className="field-help">{character} · {location} · 이미지 {images.length.toLocaleString()}장</span>
      <button type="button" className="button" disabled={action.busy} onClick={assets.reload}><RefreshCw size={14} />새로고침</button>
      {allowUpload && <>
        <button type="button" className="button primary" disabled={action.busy || readOnly} onClick={() => uploadInput.current?.click()}><Upload size={14} />에셋 업로드</button>
        <input ref={uploadInput} hidden type="file" accept="image/*" multiple onChange={event => void upload(event.target.files)} />
      </>}
    </div>
    <IllustrationFeedback error={action.error || assets.error} message={action.message} />
    {readOnly && <p className="field-help">배포 봇은 읽기 전용입니다. 배포 풀기 후 이미지를 삭제할 수 있습니다.</p>}
    {!assets.data && !assets.error ? <p className="field-help" role="status">이미지를 불러오는 중…</p>
      : assets.error ? <IllustrationLoading error={assets.error} retry={assets.reload} />
      : !images.length ? <Empty title="이미지가 없습니다">{allowUpload ? "에셋 업로드로 참조 이미지를 추가하세요." : "이 폴더에 이미지가 없습니다."}</Empty>
      : <div className="asset-gallery-grid asset-gallery-images">
        {images.map(image => <div className="asset-image-card" key={image.filename}>
          <button type="button" className="asset-image" disabled={action.busy} onClick={() => setPreview(image)}>
            <span className="asset-image-preview"><img loading="lazy" src={api.media(image.url)} alt={image.filename} /></span>
            <span className="asset-image-caption"><span className="asset-image-filename" title={image.filename}>{image.filename}</span></span>
          </button>
          <div className="asset-image-actions">
            <button type="button" className="button danger" disabled={action.busy || readOnly} onClick={() => setDeletingImage(image)}><Trash2 size={14} />삭제</button>
          </div>
        </div>)}
      </div>}
    {preview && <Modal title={`${location} · ${preview.filename}`} className="illust-character-modal" backgroundable={false} onClose={() => setPreview(null)}>
      <img className="illust-asset-preview" src={api.media(preview.url)} alt={preview.filename} />
      <div className="illust-actions">
        <button type="button" className="button danger" disabled={action.busy || readOnly} onClick={() => setDeletingImage(preview)}><Trash2 size={14} />삭제</button>
        <button type="button" className="button" disabled={action.busy} onClick={() => setPreview(null)}>닫기</button>
      </div>
    </Modal>}
    {deletingImage && <Modal title="에셋 삭제" backgroundable={false} closeDisabled={action.busy} onClose={() => setDeletingImage(null)}>
      <div className="illust-form"><p><strong>{deletingImage.filename}</strong> 파일을 삭제합니다. 되돌릴 수 없습니다.</p>
        <IllustrationFeedback error={action.error} />
        <div className="illust-actions"><button type="button" className="button" disabled={action.busy} onClick={() => setDeletingImage(null)}>취소</button><button type="button" className="button danger" disabled={action.busy} onClick={() => void removeImage(deletingImage)}>삭제 확인</button></div>
      </div>
    </Modal>}
  </div>;
}

export function EmotionFolderImagesModal({ title, onClose, ...props }: Props & { title: string; onClose: () => void }) {
  return <Modal title={title} className="illust-character-modal" backgroundable={false} onClose={onClose}>
    <EmotionFolderImages {...props} />
  </Modal>;
}
