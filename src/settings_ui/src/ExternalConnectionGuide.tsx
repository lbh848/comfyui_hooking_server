import workflow from "./externalConnectionWorkflow.json";

const workflowJson = JSON.stringify(workflow, null, 2);

// Share the connection guide and workflow across external connection pages.
export function ExternalConnectionGuide({ url }: { url: string }) {
  return (
<div className="share-guide">
                    <h3 className="sg-title">개조 삽화 모듈 설치 및 연결 가이드</h3>
                    <p className="sg-desc">RisuAI에서 개조 삽화 모듈과 Soya Comfy 플러그인을 사용하기 위한 설정 순서입니다. 다운로드부터 ComfyUI 주소 연결까지 차례대로 진행해주세요.</p>

                    <div className="sg-section">
                        <div className="sg-section-head">
                            <div className="sg-num">1</div>
                            <div>
                                <h4>모듈과 플러그인 다운로드</h4>
                                <p className="sg-sub">각 주소에서 필요한 파일을 내려받은 뒤 RisuAI에서 활성화합니다.</p>
                            </div>
                        </div>
                        <div className="sg-card">
                            <span className="sg-tag">Comfy공방 폴더</span>
                            <h5>개조 삽화 모듈 및 플러그인</h5>
                            <p>아래 Proton Drive 주소에 들어간 뒤 <strong>Comfy공방 폴더</strong>의 파일을 다운로드해주세요.</p>
                            <a className="sg-link" href="https://drive.proton.me/urls/VCVXPXN18M#pU3alCDwURvh" target="_blank" rel="noopener noreferrer">https://drive.proton.me/urls/VCVXPXN18M#pU3alCDwURvh</a>
                        </div>
                        <div className="sg-card">
                            <span className="sg-tag">백엔드 5.0.1</span>
                            <h5>라이트보드 백엔드 5.0.1</h5>
                            <p>소개 글에서 <strong>라이트보드 백엔드 5.0.1</strong>을 다운로드해주세요. 삽화 모듈은 위의 개조 삽화 모듈을 사용합니다.</p>
                            <a className="sg-link" href="https://arca.live/b/characterai/187530446" target="_blank" rel="noopener noreferrer">라이트보드 백엔드 다운로드</a>
                        </div>
                        <div className="sg-notice">다운로드한 모듈과 플러그인을 RisuAI에 등록한 뒤, 설정에서 모두 켜져 있는지 먼저 확인해주세요.</div>
                    </div>

                    <div className="sg-section">
                        <div className="sg-section-head">
                            <div className="sg-num">2</div>
                            <div>
                                <h4>RisuAI 이미지 생성 설정</h4>
                                <p className="sg-sub">기타봇의 이미지 생성 메뉴에서 ComfyUI 연결 정보를 입력합니다.</p>
                            </div>
                        </div>
                        <div className="sg-step">
                            <div className="sg-step-idx">2</div>
                            <div>
                                <h6>이미지 생성 메뉴로 이동</h6>
                                <p>RisuAI 설정에서 다음 순서로 들어가세요.</p>
                                <div className="sg-path"><span>기타봇</span><b>›</b><span>이미지 생성</span></div>
                            </div>
                        </div>
                        <div className="sg-step">
                            <div className="sg-step-idx">3</div>
                            <div>
                                <h6>이미지 생성 공급자 선택</h6>
                                <p>이미지 생성 공급자를 <strong>Comfy</strong>로 설정해주세요.</p>
                            </div>
                        </div>
                        <div className="sg-step">
                            <div className="sg-step-idx">4</div>
                            <div>
                                <h6>ComfyUI 요청 URL 입력</h6>
                                <p><strong>ComfyUI 요청 URL</strong> 항목에 발급받은 HTTPS 주소를 입력해주세요.</p>
                                <div className="sg-field"><div className="lab">ComfyUI 요청 URL</div><div className="val">{url}</div></div>
                            </div>
                        </div>
                        <div className="sg-step">
                            <div className="sg-step-idx">5</div>
                            <div>
                                <h6>Workflow 입력</h6>
                                <p>아래 Workflow JSON을 복사해서 붙여넣으세요. 이미 입력했다면 이 단계는 건너뛰어도 됩니다.</p>
                                <label className="sg-wf-label" htmlFor="external-workflow">Workflow JSON</label>
                                <textarea className="sg-wf" id="external-workflow" readOnly spellCheck={false} value={workflowJson}></textarea>
                            </div>
                        </div>
                        <div className="sg-step">
                            <div className="sg-step-idx">6</div>
                            <div>
                                <h6>Timeout 늘리기</h6>
                                <p>이미지 생성 시간이 길어질 수 있으므로 Timeout을 <strong>12000</strong> 정도로 넉넉하게 설정해주세요.</p>
                                <div className="sg-field"><div className="lab">Timeout</div><div className="val">12000</div></div>
                            </div>
                        </div>
                        <img className="sg-img" src="/api/illust/connection-guide/img1.jpg" alt="RisuAI 이미지 생성 설정 화면" />
                    </div>

                    <div className="sg-section">
                        <div className="sg-section-head">
                            <div className="sg-num">3</div>
                            <div>
                                <h4>Soya Comfy 플러그인 연결</h4>
                                <p className="sg-sub">플러그인 설정에서도 같은 서버 주소를 입력하고 연결 상태를 확인합니다.</p>
                            </div>
                        </div>
                        <div className="sg-step">
                            <div className="sg-step-idx">8</div>
                            <div>
                                <h6>Soya Comfy 플러그인 열기</h6>
                                <p>RisuAI의 플러그인 목록에서 <strong>Soya Comfy</strong> 플러그인 설정 화면으로 들어가세요.</p>
                            </div>
                        </div>
                        <div className="sg-step">
                            <div className="sg-step-idx">9</div>
                            <div>
                                <h6>서버 HTTPS 주소 입력</h6>
                                <p><strong>서버 HTTPS 주소</strong>에 앞에서 사용한 발급 주소를 동일하게 입력해주세요.</p>
                                <div className="sg-field"><div className="lab">서버 HTTPS 주소</div><div className="val">{url}</div></div>
                                <p style={{ marginTop: 10 }}>주소를 입력한 뒤 <strong>저장 및 연결 확인</strong> 버튼을 눌러 정상적으로 연결되는지 확인하세요.</p>
                            </div>
                        </div>
                        <img className="sg-img" src="/api/illust/connection-guide/img2.jpg" alt="Soya Comfy 플러그인 연결 설정 화면" />
                        <div className="sg-checklist">
                            <div className="sg-check-item">모듈과 플러그인이 활성화되어 있음</div>
                            <div className="sg-check-item">이미지 공급자가 Comfy로 설정됨</div>
                            <div className="sg-check-item">두 설정에 같은 HTTPS 주소가 입력됨</div>
                            <div className="sg-check-item">저장 및 연결 확인이 성공함</div>
                        </div>
                    </div>
                </div>
  );
}
