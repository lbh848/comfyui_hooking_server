import { useMemo, useState } from "react";
import { RefreshCw } from "lucide-react";
import { Panel } from "./components";
import { SearchableSelect } from "./SearchableSelect";
import { IllustrationFeedback } from "./IllustBackups";
import { CharacterImportedLore } from "./IllustCharacterLore";
import { LoreNpcCardDetails } from "./IllustLoreNpcImport";
import type { CharacterLoreEntry } from "./characterLore";
import type { LoreImageGroup, LoreNpc } from "./loreNpcShared";

export function IllustLoreImageReview({
  candidates,
  loreEntries,
  groups,
  groupAssignments,
  excludedFileIndexes,
  existingNames,
  selectedMatched,
  selectedImageless,
  npcResults,
  npcErrors,
  disabled,
  hasFiles,
  onGroupChange,
  onToggleMatched,
  onToggleAllMatched,
  onToggleImageless,
  onToggleAllImageless,
  onReclassify,
}: {
  candidates: LoreNpc[];
  loreEntries: CharacterLoreEntry[];
  groups: LoreImageGroup[];
  groupAssignments: Record<number, string>;
  excludedFileIndexes: ReadonlySet<number>;
  existingNames: ReadonlySet<string>;
  selectedMatched: string[];
  selectedImageless: string[];
  npcResults: ReadonlyMap<string, string>;
  npcErrors: ReadonlyMap<string, string>;
  disabled: boolean;
  hasFiles: boolean;
  onGroupChange: (groupId: number, character: string) => void;
  onToggleMatched: (name: string, checked: boolean) => void;
  onToggleAllMatched: (checked: boolean) => void;
  onToggleImageless: (name: string, checked: boolean) => void;
  onToggleAllImageless: (checked: boolean) => void;
  onReclassify?: () => void;
}) {
  const [search, setSearch] = useState("");
  const needle = search.trim().toLowerCase();
  const registered = useMemo(() => new Set(Array.from(existingNames, name => name.toLowerCase())), [existingNames]);

  const activeCountByGroup = useMemo(() => new Map(groups.map(group => [
    group.id,
    group.indexes.filter(index => !excludedFileIndexes.has(index)).length,
  ])), [groups, excludedFileIndexes]);

  const groupsByCharacter = useMemo(() => {
    const map = new Map<string, LoreImageGroup[]>();
    for (const candidate of candidates) map.set(candidate.name, []);
    for (const group of groups) {
      const assigned = groupAssignments[group.id] ?? (group.character || "");
      if (assigned && map.has(assigned)) map.get(assigned)!.push(group);
    }
    return map;
  }, [candidates, groups, groupAssignments]);

  const unmatchedGroups = useMemo(() => groups.filter(group => {
    const assigned = groupAssignments[group.id] ?? (group.character || "");
    return !assigned || !groupsByCharacter.has(assigned);
  }), [groups, groupAssignments, groupsByCharacter]);

  const matchedCandidates = useMemo(() => candidates.filter(candidate => {
    const assigned = groupsByCharacter.get(candidate.name) || [];
    return assigned.some(group => (activeCountByGroup.get(group.id) || 0) > 0);
  }), [candidates, groupsByCharacter, activeCountByGroup]);

  const imagelessCandidates = useMemo(() => candidates.filter(candidate => {
    const assigned = groupsByCharacter.get(candidate.name) || [];
    return !assigned.some(group => (activeCountByGroup.get(group.id) || 0) > 0);
  }), [candidates, groupsByCharacter, activeCountByGroup]);

  const matchedImageCount = useMemo(() => matchedCandidates.reduce((sum, candidate) => {
    const assigned = groupsByCharacter.get(candidate.name) || [];
    return sum + assigned.reduce((groupSum, group) => groupSum + (activeCountByGroup.get(group.id) || 0), 0);
  }, 0), [matchedCandidates, groupsByCharacter, activeCountByGroup]);

  const unmatchedFileCount = useMemo(
    () => unmatchedGroups.reduce((sum, group) => sum + group.count, 0),
    [unmatchedGroups],
  );

  const selectedMatchedCount = matchedCandidates.filter(candidate => selectedMatched.includes(candidate.name)).length;
  const selectableImageless = imagelessCandidates.filter(
    candidate => !registered.has(candidate.name.toLowerCase()) && !npcResults.has(candidate.name),
  );
  const selectedImagelessCount = selectableImageless.filter(candidate => selectedImageless.includes(candidate.name)).length;

  const matchesCandidate = (candidate: LoreNpc, assignedGroups: LoreImageGroup[]) => {
    if (!needle) return true;
    if (candidate.name.toLowerCase().includes(needle)) return true;
    if (candidate.aliases.some(alias => alias.toLowerCase().includes(needle))) return true;
    return assignedGroups.some(group =>
      group.group.toLowerCase().includes(needle)
      || group.samples.some(sample => sample.toLowerCase().includes(needle)),
    );
  };

  const matchesGroup = (group: LoreImageGroup) => {
    if (!needle) return true;
    return group.group.toLowerCase().includes(needle)
      || group.samples.some(sample => sample.toLowerCase().includes(needle))
      || group.reason.toLowerCase().includes(needle);
  };

  const visibleMatched = matchedCandidates.filter(candidate => matchesCandidate(candidate, groupsByCharacter.get(candidate.name) || []));
  const visibleUnmatched = unmatchedGroups.filter(matchesGroup);
  const visibleImageless = imagelessCandidates.filter(candidate => matchesCandidate(candidate, []));

  const renderGroupRow = (group: LoreImageGroup, currentCharacter: string) => {
    const activeCount = activeCountByGroup.get(group.id) || 0;
    const excludedCount = group.count - activeCount;
    return <div className="illust-lore-group-row" key={group.id}>
      <div className="illust-lore-group-meta">
        <div>
          <strong>{group.group}</strong>{" "}
          <span className="field-help">
            {activeCount.toLocaleString()}장{excludedCount > 0 ? ` (${excludedCount.toLocaleString()}장 제외됨)` : ""}
          </span>
        </div>
        {group.samples.length > 0 && <p className="field-help">예시: {group.samples.slice(0, 4).join(", ")}</p>}
        {group.reason && <p className="field-help">분류 근거: {group.reason}</p>}
      </div>
      <label className="field illust-lore-group-select">
        <span className="sr-only">{group.group} 묶음 대상 캐릭터</span>
        <SearchableSelect
          aria-label={`${group.group} 묶음 대상 캐릭터`}
          value={currentCharacter}
          disabled={disabled}
          onChange={event => onGroupChange(group.id, event.target.value)}
        >
          <option value="">미분류 (가져오지 않음)</option>
          {candidates.map(candidate => (
            <option key={candidate.name} value={candidate.name} data-search={candidate.aliases.join(" ")}>
              {candidate.name}{candidate.aliases.length ? ` (${candidate.aliases.slice(0, 2).join(", ")})` : ""}
            </option>
          ))}
        </SearchableSelect>
      </label>
    </div>;
  };

  return <Panel
    title="로어북 기반 분류 결과"
    aside={<span className="badge">일반 {matchedCandidates.length}명 · NPC {imagelessCandidates.length}명 · 미분류 {unmatchedFileCount.toLocaleString()}장</span>}
    className="illust-character-import-group"
  >
    <div className="illust-form illust-character-import-body">
      <div className="illust-image-toolbar">
        <label className="field illust-lore-search">인물·이미지 묶음 검색
          <input value={search} onChange={event => setSearch(event.target.value)} placeholder="이름, 별칭 또는 파일명" />
        </label>
        {hasFiles && candidates.length > 0 && onReclassify && (
          <button type="button" className="button secondary" disabled={disabled} onClick={onReclassify}>
            <RefreshCw size={14} />이미지 다시 분류
          </button>
        )}
      </div>

      {matchedCandidates.length > 0 && <section className="illust-lore-connection-group" data-connected="true">
        <div className="illust-lore-section-head">
          <h3>
            이미지가 매칭된 캐릭터 · 일반 캐릭터 등록
            <span className="badge">{selectedMatchedCount}/{matchedCandidates.length}명 선택 · 이미지 {matchedImageCount.toLocaleString()}장</span>
          </h3>
          <div className="illust-actions">
            <button type="button" className="button secondary small" disabled={disabled || selectedMatchedCount === matchedCandidates.length}
              onClick={() => onToggleAllMatched(true)}>모두 선택</button>
            <button type="button" className="button secondary small" disabled={disabled || selectedMatchedCount === 0}
              onClick={() => onToggleAllMatched(false)}>모두 해제</button>
          </div>
        </div>
        <p className="field-help">선택한 캐릭터는 일반 캐릭터로 등록되어 매칭된 이미지, 로어북 외형 카드(태그·별칭·선택 기준)와 근거 로어북을 저장합니다. 기존 외형 카드가 있는 캐릭터는 기존 카드를 유지하며, 대표 이미지는 자동 생성하지 않습니다.</p>
        <div className="illust-lore-list">
          {visibleMatched.map(candidate => {
            const { name } = candidate;
            const assignedGroups = groupsByCharacter.get(name) || [];
            const imageCount = assignedGroups.reduce((sum, group) => sum + (activeCountByGroup.get(group.id) || 0), 0);
            const checked = selectedMatched.includes(name);
            const isExisting = registered.has(name.toLowerCase());
            return <div className="illust-lore-entry" key={name}>
              <div className="illust-lore-candidate-head">
                <label className="illust-check">
                  <input type="checkbox" disabled={disabled} checked={checked}
                    onChange={event => onToggleMatched(name, event.target.checked)} />
                  <strong>{name}</strong>
                </label>
                <div className="illust-actions">
                  <span className="badge">{isExisting ? "기존 캐릭터" : "새 일반 캐릭터"}</span>
                  <span className="badge">이미지 {imageCount.toLocaleString()}장</span>
                  <span className="field-help">로어북 {candidate.entry_indexes.length}개</span>
                </div>
              </div>
              {candidate.aliases.length > 0 && <p className="field-help">{candidate.aliases.join(" · ")}</p>}
              {candidate.description && <p className="illust-lore-text">{candidate.description}</p>}
              <LoreNpcCardDetails card={candidate.card} />
              <details className="illust-lore-entry">
                <summary>배정된 이미지 묶음 {assignedGroups.length}개 · {imageCount.toLocaleString()}장</summary>
                <div className="illust-form illust-lore-content">
                  {assignedGroups.map(group => renderGroupRow(group, name))}
                </div>
              </details>
              <CharacterImportedLore entries={candidate.entry_indexes.map(index => loreEntries[index]).filter(Boolean)} />
            </div>;
          })}
          {!visibleMatched.length && <p className="field-help">검색 결과와 일치하는 캐릭터가 없습니다.</p>}
        </div>
      </section>}

      {unmatchedGroups.length > 0 && <section className="illust-lore-connection-group" data-connected="false">
        <details open={matchedCandidates.length === 0 || !!needle || undefined}>
          <summary>
            <strong>미분류 이미지 (가져오기 제외)</strong>{" "}
            <span className="badge">{unmatchedGroups.length}묶음 · {unmatchedFileCount.toLocaleString()}장</span>
          </summary>
          <div className="illust-form illust-lore-content">
            <p className="field-help">배경·아이콘·비인물 이미지이거나 로어북 인물에 매칭되지 않은 이미지입니다. 필요한 묶음만 대상 캐릭터를 지정해 포함할 수 있습니다.</p>
            <div className="illust-lore-list">
              {visibleUnmatched.map(group => renderGroupRow(group, ""))}
              {!visibleUnmatched.length && <p className="field-help">검색 결과와 일치하는 미분류 묶음이 없습니다.</p>}
            </div>
          </div>
        </details>
      </section>}

      {imagelessCandidates.length > 0 && <section className="illust-lore-connection-group" data-connected="true">
        <div className="illust-lore-section-head">
          <h3>
            이미지 없는 인물 · 로어북 기반 NPC 생성
            <span className="badge">{selectedImagelessCount}명 생성 예정</span>
          </h3>
          {selectableImageless.length > 0 && <div className="illust-actions">
            <button type="button" className="button secondary small" disabled={disabled || selectedImagelessCount === selectableImageless.length}
              onClick={() => onToggleAllImageless(true)}>모두 선택</button>
            <button type="button" className="button secondary small" disabled={disabled || selectedImagelessCount === 0}
              onClick={() => onToggleAllImageless(false)}>모두 해제</button>
          </div>}
        </div>
        <p className="field-help">매칭된 이미지가 없는 인물입니다. 체크하면 기존 로어북 기반 NPC 생성과 동일하게 외형 카드로 대표 이미지를 생성하고 수명을 잠급니다.</p>
        <div className="illust-lore-list">
          {visibleImageless.map(candidate => {
            const { name } = candidate;
            const status = npcResults.get(name) || (registered.has(name.toLowerCase()) ? "이미 등록됨" : "");
            return <div className="illust-lore-entry" key={name}>
              <label className="illust-check">
                <input type="checkbox" disabled={disabled || !!status} checked={!status && selectedImageless.includes(name)}
                  onChange={event => onToggleImageless(name, event.target.checked)} />
                <strong>{name}</strong>{status && <span>{status}</span>}
              </label>
              {candidate.aliases.length > 0 && <p className="field-help">{candidate.aliases.join(" · ")}</p>}
              <p className="illust-lore-text">{candidate.description}</p>
              <LoreNpcCardDetails card={candidate.card} />
              {!status && npcErrors.has(name) && <IllustrationFeedback error={npcErrors.get(name)} />}
              <CharacterImportedLore entries={candidate.entry_indexes.map(index => loreEntries[index]).filter(Boolean)} />
            </div>;
          })}
          {!visibleImageless.length && <p className="field-help">검색 결과와 일치하는 이미지 없는 인물이 없습니다.</p>}
        </div>
      </section>}
    </div>
  </Panel>;
}
