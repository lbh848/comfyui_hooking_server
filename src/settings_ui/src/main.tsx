import React from "react";
import ReactDOM from "react-dom/client";
import "@fontsource-variable/noto-sans-kr";
import App from "./App";
import IllustApp from "./IllustApp";
import EmotionApp from "./EmotionApp";
import AssetApp from "./AssetApp";
import LoraApp from "./LoraApp";
import { ErrorBoundary } from "./ErrorBoundary";
import { ModalWorkspace } from "./ModalWorkspace";
import { WorkshopNavigation, WorkshopScreen } from "./WorkshopNavigation";
import { captureBrowserErrors } from "./programLogs";
import "./styles.css";

captureBrowserErrors();

// Release numeric input focus before the browser handles a wheel event.
// Leave the event passive so scrolling the page or a dialog still works.
document.addEventListener("wheel", (event) => {
  const input = event.target;
  if (input instanceof HTMLInputElement && input.type === "number" && document.activeElement === input) {
    input.blur();
  }
}, { capture: true, passive: true });

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <ErrorBoundary>
      <WorkshopNavigation><ModalWorkspace>
        <WorkshopScreen workshop="setting"><App /></WorkshopScreen>
        <WorkshopScreen workshop="illust"><IllustApp /></WorkshopScreen>
        <WorkshopScreen workshop="emotion"><EmotionApp /></WorkshopScreen>
        <WorkshopScreen workshop="asset"><AssetApp /></WorkshopScreen>
        <WorkshopScreen workshop="lora"><LoraApp /></WorkshopScreen>
      </ModalWorkspace></WorkshopNavigation>
    </ErrorBoundary>
  </React.StrictMode>,
);
