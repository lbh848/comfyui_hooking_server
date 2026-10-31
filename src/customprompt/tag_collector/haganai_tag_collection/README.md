# haganai_tag_collection

하가나이(Boku wa Tomodachi ga Sukunai) 캐릭터별 태깅 데이터 수집 결과물.

데이터셋 원본 경로의 `.txt` 태깅 파일들을 읽어 NSFW 파일을 제외하고,
캐릭터별로 중복 제거된 고유 태그 목록을 생성한 것이다.

이 결과물을 바탕으로 `haganai_extract_outfit_prompt.py`의 `CHAR_IDENTITIES` 딕셔너리를
실제 태깅 데이터 기반으로 강화했다.

---

## 폴더 구성

```
haganai_tag_collection/
├── README.md                          ← 본 파일
├── collect_tags.py                    ← 태그 수집 스크립트
├── tags_cleaned_hasegawa_kobato.txt   ← 코바토 고유 태그
├── tags_cleaned_hasegawa_kodaka.txt   ← 코다카 고유 태그
├── tags_cleaned_hidaka_hinata.txt     ← 히나타 고유 태그
├── tags_cleaned_jinguuji_karin.txt    ← 카린 고유 태그
├── tags_cleaned_kashiwazaki_pegasus.txt ← 페가수스 고유 태그
├── tags_cleaned_kashiwazaki_sena.txt  ← 세나 고유 태그
├── tags_cleaned_kusunoki_yukimura.txt ← 유키무라 고유 태그
├── tags_cleaned_mikazuki_yozora.txt   ← 요조라 고유 태그
├── tags_cleaned_ohtomo_akane.txt      ← 아카네 고유 태그
├── tags_cleaned_oreki_houtarou.txt    ← 호타로 고유 태그
├── tags_cleaned_redfield_stella.txt   ← 스텔라 고유 태그
├── tags_cleaned_shiguma_rika.txt      ← 리카 고유 태그
├── tags_cleaned_takayama_kate.txt     ← 케이트 고유 태그
├── tags_cleaned_takayama_maria.txt    ← 마리아 고유 태그
└── tags_cleaned_yusa_aoi.txt          ← 아오이 고유 태그
```

---

## 파일 설명

### collect_tags.py

데이터셋 경로의 캐릭터별 폴더에서 `.txt` 태깅 파일을 읽어 처리하는 스크립트.

**처리 과정:**
1. 캐릭터 폴더 내 `.txt` 파일 스캔
2. 파일명에 NSFW 태그가 포함된 파일 제외
3. 쉼표로 분리된 태그를 모두 수집
4. 대소문자 무시 중복 제거 (순서 유지)
5. `tags_cleaned_{캐릭터명}.txt`로 저장

**NSFW 제외 태그 목록 (33개):**
breast caress, cowgirl back cumshot, cowgirl back, cowgirl cumshot, cowgirl,
doggystyle cumshot, doggystyle, fellatio, fingering, footjob cumshot, footjob,
fullnelson cumshot, fullnelson, handjob, masturbation, mating press cumshot,
mating press, missionary position cumshot, missionary position, paizuri,
reverse pright straddle cumshot, reverse pright straddle, reverse standing position,
showing armpit, showing nude, spooning cumshot, spooning, standing position,
suspended congress cumshot, suspended congress, upright straddle cumshot, upright straddle

**실행 방법:**
```bash
cd customprompt/haganai_tag_collection
python collect_tags.py
```

### tags_cleaned_*.txt

각 캐릭터의 중복 제거된 고유 태그가 쉼표로 구분되어 저장된 텍스트 파일.
형식: `태그1, 태그2, 태그3, ...`

---

## 수집 통계

| 캐릭터 | 읽은 파일 | NSFW 제외 | 고유 태그 수 |
|---|---|---|---|
| hasegawa kobato | 37 | 32 | 123 |
| kashiwazaki sena | 37 | 32 | 132 |
| kusunoki yukimura | 37 | 32 | 134 |
| mikazuki yozora | 37 | 32 | 144 |
| shiguma rika | 37 | 32 | 137 |
| takayama kate | 37 | 32 | 138 |
| takayama maria | 37 | 32 | 153 |
| redfield stella | 37 | 32 | 116 |
| yusa aoi | 37 | 32 | 138 |
| oreki houtarou | 9 | 0 | 52 |
| hasegawa kodaka | 1 | 0 | 35 |
| hidaka hinata | 1 | 0 | 36 |
| jinguuji karin | 1 | 0 | 36 |
| ohtomo akane | 1 | 0 | 36 |
| kashiwazaki pegasus | 1 | 0 | 28 |

---

## 활용 결과

수집된 태그에서 복장 관련 태그를 추출하여
`haganai_extract_outfit_prompt.py`의 `CHAR_IDENTITIES`를 강화함.

- 기존 9명 → 15명으로 확장
- 각 캐릭터의 복장 정보를 실제 태깅 데이터 기반으로 상세화
