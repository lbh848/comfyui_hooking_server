import { useState } from "react";
import { CircleHelp } from "lucide-react";
import { Modal, Toggle } from "./components";

export function AssetGalleryKeyboardTools({ enabled, onChange }: { enabled: boolean; onChange: (enabled: boolean) => void }) {
  const [help, setHelp] = useState(false);
  return <>
    <div className="asset-generation-tag-row asset-keyboard-tools" role="group" aria-label="Tools">
      <span className="asset-generation-tag-label">Tools</span>
      <div className="asset-keyboard-toggle"><span>키보드 조작 활성화</span><Toggle label="키보드 조작 활성화" checked={enabled} onChange={onChange} /></div>
      <button type="button" className="icon-button" aria-label="키보드 조작 도움말" title="키보드 조작 도움말" onClick={() => setHelp(true)}><CircleHelp size={17} /></button>
    </div>
    {help && <Modal title="키보드 조작 도움말" className="asset-keyboard-help" backgroundable={false} onClose={() => setHelp(false)}>
      <p className="field-help">키보드 조작을 켜면 목록의 선택 항목에 테두리가 표시됩니다. LV0는 캐릭터, LV1은 복장·표정 폴더, LV2는 이미지 목록입니다.</p>
      <dl className="asset-keyboard-shortcuts">
        <div><dt>방향키</dt><dd>목록의 항목을 선택합니다. 라이트뷰어에서는 이전·다음 이미지를 봅니다.</dd></div>
        <div><dt><kbd>Enter</kbd></dt><dd>선택한 폴더에 진입합니다. LV2에서는 선택한 이미지의 라이트뷰어를 엽니다.</dd></div>
        <div><dt><kbd>Esc</kbd></dt><dd>한 단계 위로 돌아갑니다. 라이트뷰어에서는 뷰어를 닫고, 삭제 확인창에서는 삭제를 취소합니다.</dd></div>
        <div><dt><kbd>D</kbd></dt><dd>선택한 이미지 또는 라이트뷰어의 이미지에 대한 삭제 확인창을 엽니다.</dd></div>
        <div><dt>삭제 확인창</dt><dd>방향키 또는 Tab으로 버튼을 이동하고 Enter로 실행합니다. 처음에는 ‘취소’가 선택됩니다.</dd></div>
        <div><dt><kbd>A</kbd></dt><dd>선택한 이미지 또는 라이트뷰어의 이미지를 대표로 설정합니다. 이미 대표인 이미지에서 다시 누르면 해제합니다.</dd></div>
      </dl>
      <p className="field-help">검색·태그 등 입력칸과 다른 창을 사용 중일 때는 목록 단축키가 동작하지 않습니다.</p>
      <div className="modal-actions"><button type="button" className="button primary" onClick={() => setHelp(false)}>닫기</button></div>
    </Modal>}
  </>;
}
