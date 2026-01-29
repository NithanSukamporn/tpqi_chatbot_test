// server.js
const express = require('express');
const cors = require('cors');
const { OpenAI } = require('openai');
const admin = require('firebase-admin');
require('dotenv').config();

const app = express();
app.use(cors()); // อนุญาตให้หน้าเว็บยิงเข้ามาได้
app.use(express.json());

// 1. เชื่อมต่อ Firebase (สำหรับ RAG)
// เทคนิค: บน Render เราจะอ่านไฟล์ JSON ผ่าน Environment Variable หรือไฟล์
// เพื่อความง่ายใน local ให้ใช้ไฟล์ แต่บน Render เราจะใช้วิธีพิเศษ (ดูขั้นตอนที่ 3)
let serviceAccount;
if (process.env.FIREBASE_CREDENTIALS) {
    serviceAccount = JSON.parse(process.env.FIREBASE_CREDENTIALS);
} else {
    serviceAccount = require('./serviceAccountKey.json');
}

admin.initializeApp({
  credential: admin.credential.cert(serviceAccount)
});
const db = admin.firestore();

// 2. ตั้งค่า OpenAI
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// 3. API Endpoint (จุดรับแขก)
app.post('/chat', async (req, res) => {
    try {
        const { message, image } = req.body;
        console.log("ได้รับข้อความ:", message);

        let contextText = "";

        // --- RAG Logic Starts Here ---
        if (message) {
            // A. แปลงคำถามเป็น Vector
            const embedding = await openai.embeddings.create({
                model: "text-embedding-3-small",
                input: message,
            });
            const vector = embedding.data[0].embedding;

            // B. ค้นหาใน Firestore
            const coll = db.collection("legal_knowledge_vectors");
            // หมายเหตุ: ต้องแน่ใจว่าสร้าง Index ใน Firebase แล้ว
            const snapshot = await coll.findNearest("vector", vector, {
                limit: 3,
                distanceMeasure: "COSINE"
            }).get();

            const docs = [];
            snapshot.forEach(doc => {
                const d = doc.data();
                docs.push(`- หัวข้อ: ${d.topic}\n  เนื้อหา: ${d.content}`);
            });
            
            if (docs.length > 0) contextText = docs.join("\n\n");
            else contextText = "ไม่พบข้อมูลในฐานข้อมูล";
        }
        // --- RAG Logic Ends ---

        // C. สร้าง Prompt
        const systemPrompt = `
        คุณคือผู้ช่วย AI กฎหมาย
        ข้อมูลอ้างอิง:
        ${contextText}
        
        คำสั่ง: ตอบคำถามโดยอ้างอิงข้อมูลด้านบนเท่านั้น หากไม่มีให้ตอบว่าไม่ทราบ
        `;

        const messages = [{ role: "system", content: systemPrompt }];
        const userContent = [];
        if (message) userContent.push({ type: "text", text: message });
        if (image) userContent.push({ type: "image_url", image_url: { url: image } });
        messages.push({ role: "user", content: userContent });

        // D. ส่ง OpenAI
        const completion = await openai.chat.completions.create({
            model: "gpt-4o-mini",
            messages: messages,
        });

        res.json({ 
            answer: completion.choices[0].message.content,
            context: contextText 
        });

    } catch (error) {
        console.error("Error:", error);
        res.status(500).json({ error: error.message });
    }
});

// เริ่ม Server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));