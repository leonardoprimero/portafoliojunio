const express = require('express');
const qrcode = require('qrcode-terminal');
const { Client, LocalAuth } = require('whatsapp-web.js');
const axios = require('axios');
const admin = require('firebase-admin');

function cleanPhone(phone) {
    return phone.replace(/[^0-9]/g, ''); // Saca todo menos los números
}


try {
    const serviceAccount = require('./serviceAccountKey.json');
    admin.initializeApp({
        credential: admin.credential.cert(serviceAccount)
    });
} catch (error) {
    console.error("❌ Error crítico: No se pudo encontrar o leer el archivo 'serviceAccountKey.json'.");
    console.error("Asegúrate de que el archivo exista en la misma carpeta que tu script.");
    process.exit(1); // Detiene la ejecución si no se puede inicializar Firebase.
}

const db = admin.firestore();
// ---------------------------------

const app = express();
app.use(express.json());

const client = new Client({
    authStrategy: new LocalAuth(),
    puppeteer: {
        args: ["--no-sandbox", "--disable-setuid-sandbox"],
    }
});

// Objeto para manejar el estado de cada cliente (buffer de mensajes y temporizador)
const estadosClientes = {};
const DEBOUNCE_MS = 15000; // 15 segundos de espera desde el último mensaje

client.on('qr', (qr) => {
    console.log("📱 Escanea este código QR desde la sección 'Dispositivos vinculados' de tu WhatsApp.");
    qrcode.generate(qr, { small: true });
});

client.on('ready', () => {
    console.log('✅ Cliente de WhatsApp conectado y listo para operar.');
});

client.on('auth_failure', msg => {
    console.error('❌ ERROR DE AUTENTICACIÓN:', msg);
});

client.initialize();

// ====== FUNCIÓN PARA GUARDAR CONTACTO E HISTORIAL DE MENSAJES ======
async function guardarContactoYMensaje({ phone, name, empresa, email, mensaje, emisor }) {
    try {
        const ref = db.collection('clientes').doc(phone);
        const doc = await ref.get();

        if (!doc.exists) {
            // Si el cliente no existe, lo crea con sus datos y el historial.
            await ref.set({
                phone,
                name, // SOLO se setea aquí (primera vez)
                empresa: empresa || '',
                email: email || '',
                creado: admin.firestore.FieldValue.serverTimestamp(),
                historial: []
            });
        } else {
            // Si ya existe, actualiza los demás campos, PERO NO el name
            let updates = {};
            // Solo actualizá empresa/email si querés, pero NO name
            if (empresa && (!doc.data().empresa || doc.data().empresa === '')) updates.empresa = empresa;
            if (email && (!doc.data().email || doc.data().email === '')) updates.email = email;
            if (Object.keys(updates).length > 0) {
                await ref.update(updates);
            }
        }

        // Añade el mensaje actual al historial del cliente.
        await ref.update({
            historial: admin.firestore.FieldValue.arrayUnion({
                fecha: new Date().toISOString(),
                mensaje,
                emisor // Puede ser 'cliente' o 'Lilith'
            })
        });
    } catch (err) {
        console.error(`❌ Error guardando en Firebase para ${phone}:`, err.message);
    }
}
// =================================================================

// ==== MANEJO DE MENSAJES ENTRANTES DE WHATSAPP ====
client.on('message', async msg => {
    if (msg.fromMe || msg.isStatus || msg.from.endsWith('@g.us')) return;

    const phone = cleanPhone(msg.from); // ¡Usá el número limpio SIEMPRE!
    const name = msg._data.notifyName || '';
    const message = msg.body;

    // 1. Guardar SIEMPRE el mensaje en Firebase de forma inmediata.
    await guardarContactoYMensaje({
        phone,
        name,
        mensaje: message,
        emisor: 'cliente'
    });

    
    // 2. Inicializar el buffer para este cliente si es su primer mensaje en la ráfaga.
    if (!estadosClientes[phone]) {
        estadosClientes[phone] = { mensajes: [] };
    }
    // Añadir el mensaje actual al buffer de mensajes en memoria.
    estadosClientes[phone].mensajes.push(message);

    // 3. Reiniciar el temporizador (debounce) cada vez que llega un mensaje nuevo.
    if (estadosClientes[phone].timeout) {
        clearTimeout(estadosClientes[phone].timeout);
    }

    // Establecer un nuevo temporizador.
    estadosClientes[phone].timeout = setTimeout(async () => {
        // Cuando el temporizador se completa, concatenar todos los mensajes del buffer.
        const mensajesAcumulados = estadosClientes[phone].mensajes.join('\n');

        try {
            // Enviar los mensajes acumulados al webhook de n8n.
            await axios.post(
                'https://auto0.leocaliva.com/webhook/lilith-disparo',
                {
                    phone: phone,
                    name: name,
                    message: mensajesAcumulados  // <-- Aquí se envían todos los mensajes juntos.
                }
             );
            console.log(`🔗 [Debounce] Mensajes reenviados a n8n para ${phone}: "${mensajesAcumulados}"`);
        } catch (err) {
            console.error('❌ [Debounce] Error enviando mensajes a n8n:', err.message);
        }

        // Limpiar el estado del cliente después de enviar para no reenviar los mismos mensajes.
        delete estadosClientes[phone];
    }, DEBOUNCE_MS);
});
// ===================================================

// --- ENDPOINTS DE LA API PARA N8N Y OTROS SERVICIOS ---

// Endpoint para que n8n envíe mensajes a través del bot.
app.post("/send", async (req, res) => {
    const { phone, message, name, empresa, email } = req.body;
    if (!phone || !message) {
        return res.status(400).send({ success: false, error: "Faltan los campos 'phone' o 'message'" });
    }
    try {
        const chatId = phone.replace(/\D/g, "") + "@c.us";

        // Enviar el mensaje de la IA (Lilith).
        await client.sendMessage(chatId, message);

        // Guardar la respuesta de la IA en el historial de Firebase.
        await guardarContactoYMensaje({
            phone,
            name,
            empresa,
            email,
            mensaje: message,
            emisor: "Lilith"
        });

        res.status(200).send({ success: true });
    } catch (error) {
        console.error("❌ Error en /send:", error.message);
        res.status(500).send({ success: false, error: "No se pudo enviar el mensaje." });
    }
});

// Endpoint para verificar si un número tiene una cuenta de WhatsApp válida.
app.post('/check-number', async (req, res) => {
    const { phone } = req.body;
    console.log(`📩 Número recibido para verificación: ${phone}`);

    if (!phone) {
        return res.status(400).json({ success: false, error: 'Falta el número de teléfono.' });
    }

    try {
        const chatId = phone.replace(/\D/g, '') + "@c.us";
        const isRegistered = await client.isRegisteredUser(chatId);
        res.status(200).json({ success: true, isRegistered });
    } catch (error) {
        console.error('❌ Error en /check-number:', error.message);
        res.status(500).json({ success: false, error: 'No se pudo verificar el número.' });
    }
});

// Endpoint de estado para verificar que el servidor está funcionando.
app.get('/', (req, res) => {
    res.send('🟢 Servidor del Bot de WhatsApp está corriendo correctamente.');
});

// === ENDPOINT PARA TRAER EL DOCUMENTO ENTERO DEL CLIENTE POR TELÉFONO ===
app.get('/historial/:phone', async (req, res) => {
    const { phone } = req.params;
    try {
        const ref = db.collection('clientes').doc(phone);
        const doc = await ref.get();
        if (!doc.exists) {
            return res.status(404).json({ error: 'Cliente no encontrado' });
        }
        const data = doc.data();
        // Mandá todos los campos importantes
        return res.json({
            name: data.name || "",
            empresa: data.empresa || "",
            email: data.email || "",
            phone: data.phone || phone,
            historial: data.historial || [],
            creado: data.creado || null
        });
    } catch (error) {
        console.error('❌ Error trayendo historial:', error.message);
        res.status(500).json({ error: 'Error consultando Firebase' });
    }
});
// === ENDPOINT PARA TRAER EL DOCUMENTO ENTERO DEL CLIENTE POR EMAIL ===
app.get('/cliente-por-email/:email', async (req, res) => {
    const { email } = req.params;
    if (!email) {
        return res.status(400).json({ error: 'Falta el email' });
    }

    try {
        // Hacemos una consulta a la colección 'clientes' para buscar por el campo 'email'
        const clientesRef = db.collection('clientes');
        const snapshot = await clientesRef.where('email', '==', email).limit(1).get();

        if (snapshot.empty) {
            // Si no se encuentra ningún cliente con ese email
            return res.status(404).json({ error: 'Cliente no encontrado con ese email' });
        }

        // Si se encuentra, tomamos el primer resultado
        const doc = snapshot.docs[0];
        const data = doc.data();

        // Devolvemos todos los datos del cliente encontrado
        return res.json({
            name: data.name || "",
            empresa: data.empresa || "",
            email: data.email || email,
            phone: data.phone || "", // Puede que no tenga teléfono si se registró por email
            historial: data.historial || [],
            creado: data.creado || null
        });

    } catch (error) {
        console.error('❌ Error buscando cliente por email:', error.message);
        res.status(500).json({ error: 'Error consultando Firebase' });
    }
});

// Endpoint para sumar mensaje al historial por email (solo email y mensaje, emisor se setea "cliente")
app.post('/add-mensaje-por-email', async (req, res) => {
    const { email, mensaje, emisor } = req.body;
    if (!email) {
        return res.status(400).json({ error: 'Falta email' });
    }
    try {
        const clientesRef = db.collection('clientes');
        const snap = await clientesRef.where('email', '==', email).limit(1).get();
        const entradaHistorial = {
            fecha: new Date().toISOString(),
            mensaje: mensaje || "", // Puede venir vacío
            emisor: emisor ||"cliente"
        };
        if (snap.empty) {
            // Si no existe el cliente, lo crea solo con email e historial
            await clientesRef.add({
                email,
                creado: admin.firestore.FieldValue.serverTimestamp(),
                historial: [entradaHistorial]
            });
            return res.json({ success: true, created: true });
        } else {
            // Si existe, suma el mensaje al historial
            const docRef = snap.docs[0].ref;
            await docRef.update({
                historial: admin.firestore.FieldValue.arrayUnion(entradaHistorial)
            });
            return res.json({ success: true, updated: true });
        }
    } catch (err) {
        console.error("❌ Error en /add-mensaje-por-email:", err.message);
        res.status(500).json({ error: 'Error consultando Firebase' });
    }
});




const PORT = process.env.PORT || 3030;
app.listen(PORT, '0.0.0.0', () => {
    console.log(`🟢 Servidor escuchando en el puerto ${PORT}.`);
});
