import { ChevronRight, FolderOpen, Star, UserRound, X } from "lucide-react";
import type { ReactNode } from "react";

export function AssetFolderDeleteButton({ label, disabled, title, onClick }: {
  label: string; disabled?: boolean; title?: string; onClick: () => void;
}) {
  return <button type="button" className="icon-button asset-folder-delete" disabled={disabled} aria-label={`${label} 삭제`} title={title || `${label} 삭제`}
    draggable={false} onPointerDown={event => event.stopPropagation()} onClick={event => { event.stopPropagation(); onClick(); }}><X size={16} /></button>;
}

export function AssetFolderButton({ label, galleryKey, previewUrl, previewAlt, count, character = false, icon, description, disabled, selected, onClick }: {
  label: string; galleryKey?: string; previewUrl?: string; previewAlt?: string; count?: number; character?: boolean; icon?: ReactNode; description?: ReactNode; disabled?: boolean; selected?: boolean; onClick: () => void;
}) {
  return <button type="button" className="asset-folder" data-asset-gallery-item={galleryKey} disabled={disabled} aria-pressed={selected} onClick={onClick}>
    <span className="asset-folder-preview">{previewUrl ? <img loading="lazy" draggable={false} src={previewUrl} alt={previewAlt || `${label} 대표 이미지`} /> : icon ?? (character ? <UserRound size={32} /> : <FolderOpen size={32} />)}</span>
    <span className="asset-folder-description"><strong>{label}</strong>{count !== undefined && <small>{count}장</small>}
      {description !== undefined ? description : previewUrl ? <span className="asset-folder-representative"><Star size={12} fill="currentColor" />대표 이미지</span> : <small>대표 이미지 없음</small>}
    </span><ChevronRight className="asset-folder-arrow" size={16} />
  </button>;
}
