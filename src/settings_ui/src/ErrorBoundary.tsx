import { Component, type ErrorInfo, type ReactNode } from "react";
import { reportProgramError } from "./programLogs";

type State = { hasError: boolean };

export class ErrorBoundary extends Component<{ children: ReactNode }, State> {
  state: State = { hasError: false };

  static getDerivedStateFromError(): State {
    return { hasError: true };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    reportProgramError(error, info.componentStack?.trim() || "React ErrorBoundary");
  }

  render() {
    if (!this.state.hasError) return this.props.children;
    return (
      <main className="app-error-boundary" role="alert">
        <div className="settings-feedback error">
          <p>화면을 표시하는 중 오류가 발생했습니다. 브라우저를 새로고침해 주세요.</p>
          <button type="button" className="button secondary" onClick={() => window.location.reload()}>
            화면 새로고침
          </button>
        </div>
      </main>
    );
  }
}
