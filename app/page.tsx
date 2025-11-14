import AudioRecorder from "./components/AudioRecorder";

export default function Home() {
  return (
    <main style={{
      display: "flex",
      justifyContent: "center",
      alignItems: "center",
      minHeight: "100vh",
      backgroundColor: "#f0f2f5",
    }}>
      <AudioRecorder />
    </main>
  );
}
