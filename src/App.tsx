import { useEffect, useRef, useState } from "react";
import * as faceapi from '@vladmandic/face-api';

const distanceThreshold = 0.55;
const modelPath = './models';

function App() {
    const videoRef = useRef<HTMLVideoElement | null>(null);
    const referenceRef = useRef<Float32Array | null>(null);
    const animationFrameIdRef = useRef<number | null>(null);
    const generationRef = useRef(0);

    const [cameraReady, setCameraReady] = useState(false);
    const [modelsLoaded, setModelsLoaded] = useState(false);
    const [matchResult, setMatchResult] = useState<string>('Ожидание...');
    const [error, setError] = useState<string | null>(null);

    async function checkMatches(generation: number) {
        if (!videoRef.current) return;

        const video = videoRef.current;
        if (video.readyState === video.HAVE_ENOUGH_DATA) {
            const detection = await faceapi.detectSingleFace(video)
                .withFaceLandmarks()
                .withFaceDescriptor();

            if(!referenceRef.current) return;
            if (detection) {
                const distance = faceapi.euclideanDistance(referenceRef.current, detection.descriptor);
                setMatchResult(distance < distanceThreshold
                    ? `Это вы! ✅ (Уверенность ИИ: ${Math.round((1 - distance) * 100)}%)`
                    : 'Это НЕ вы! ❌'
                )
            }
        }

        if (generationRef.current === generation) {
            animationFrameIdRef.current = requestAnimationFrame(() => checkMatches(generation));
        }
    }

    // делаем снимок и запускаем цикл проверки, если еще не запущен
    async function takeSnapshot() {
        if (!videoRef.current) return;
        if (animationFrameIdRef.current) {
            cancelAnimationFrame(animationFrameIdRef.current);
            animationFrameIdRef.current = null;
            referenceRef.current = null;
            generationRef.current++;
        }

        const detection = await faceapi.detectSingleFace(videoRef.current)
            .withFaceLandmarks()
            .withFaceDescriptor();

        if (detection) {
            referenceRef.current = detection.descriptor;
            if (!animationFrameIdRef.current) {
                animationFrameIdRef.current = requestAnimationFrame(() => checkMatches(generationRef.current));
            }
        } else {
            setMatchResult('Лицо не найдено');
        }
    }

    // инициализация камеры и моделей анализа изображений
    useEffect(() => {
        let stream: MediaStream | null = null;
        let cancelled = false;
        // инициализация камеры  видео
        (async function() {
            try {
                stream = await navigator.mediaDevices.getUserMedia({
                    video: { width: 640, height: 480 },
                    audio: false,
                })
                if (cancelled) {
                    stream.getTracks().forEach(track => track.stop());
                    return;
                }
                if (videoRef.current) {
                    videoRef.current.srcObject = stream;
                    await videoRef.current.play();
                    setCameraReady(true);
                }
            } catch (e: any) {
                setError('Не удалось получить доступ к камере: ' + e.message);
            }
        })();
        // загрузка моделей
        (async function() {
            try {
                await faceapi.nets.ssdMobilenetv1.loadFromUri(modelPath);
                await faceapi.nets.faceLandmark68Net.loadFromUri(modelPath);
                await faceapi.nets.faceRecognitionNet.loadFromUri(modelPath);
                setModelsLoaded(true);
            } catch (err) {
                setError('Ошибка загрузки модели: ' + (err as Error).message);
            }
        })();

        return () => {
            cancelled = true;
            if (stream) stream.getTracks().forEach(track => track.stop());
            if (animationFrameIdRef.current) cancelAnimationFrame(animationFrameIdRef.current);
        }
    }, []);

    return <div
        style={{
            textAlign: 'center', position: 'relative', width: 'fit-content', height: 'fit-content', margin: "auto",
        }}
    >
        {error && <p style={{ color: 'red' }}>{error}</p>}
        {!cameraReady && !error && <p>Запуск камеры...</p>}
        {!modelsLoaded && !error && <p>Качаем модели...</p>}

        <video ref={videoRef} width={640} height={480} />
        <div style={{ marginTop: '10px' }}>
            {cameraReady && modelsLoaded && (
                <button onClick={takeSnapshot} style={{ padding: '10px 20px', fontSize: '16px' }}>
                    Запомнить меня
                </button>
            )}
            <p style={{ fontSize: '20px', fontWeight: 'bold' }}>{matchResult}</p>
        </div>
    </div>
}

export default App
