      const ESP_URL = "http://localhost:9080";

      // Оновлення UI для режимів

      function updateUI(mode, startTime, endTime) {
        const isAuto = (mode === "AUTO");
        document.getElementById("modeDisplay").textContent = `Mode: ${mode}`;

        // Показати/приховати елементи
        document.getElementById("toggleBtn").classList.toggle("hidden", isAuto);
        document.getElementById("autoSettings").classList.toggle("hidden", !isAuto);

       // Якщо режим AUTO і час 00:00 - 00:00, Виводимо "no timer set"
        if (isAuto && startTime === "00:00" && endTime === "00:00") {
          console.log("no timer set");
        }
      }

     
        // Функція синхронізації : Отримати в якому стані зараз світло
       
      async function synchronize() {
        try {
          const res = await fetch(`${ESP_URL}/info`);
          if (!res.ok) throw new Error("Bad response");
          const data = await res.json();

          // Встановлення часу 
          document.getElementById("info").textContent = `Light: ${data.relay}`;
          document.getElementById("startTime").value = data.start_time || "";
          document.getElementById("endTime").value   = data.end_time   || "";

          // слайдер та UI
          const modeSwitch = document.getElementById("modeSwitch");
          const isAuto = (data.mode === "AUTO");
          modeSwitch.checked = isAuto;
          updateUI(data.mode, data.start_time, data.end_time);

        } catch (err) {
          // Помилка, якщо не вдалось підключитись
          alert("No connection. Try again");
        }
      }

		// Ввімкнення та вимкнення світла (реле) в ручному режимі
      async function toggleRelay() {
        try {
          await fetch(`${ESP_URL}/toggle`);
          await synchronize();
        } catch (err) {
          alert("No connection. Try again");
        }
      }

	// Встановлення автономного режиму з заданим часом
      async function setAutoMode() {
        const now = new Date();
        const currentTime = now.getHours() * 3600 + now.getMinutes() * 60;
        const startTimeStr = document.getElementById("startTime").value;
        const endTimeStr   = document.getElementById("endTime").value;

        if (!startTimeStr || !endTimeStr) {
          alert("Please enter valid start and end times!");
          return;
        }

        const data = {
          current_time: currentTime,
          start_time: timeToSeconds(startTimeStr),
          end_time: timeToSeconds(endTimeStr)
        };

        try {
          await fetch(`${ESP_URL}/automatic_mode`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data)
          });
          await synchronize();
        } catch (err) {
          alert("No connection. Try again");
        }
      }

     
	// Перемикаємо режим тільки, якщо сервер відповів "OK"
       
      async function changeMode() {
        const modeSwitch = document.getElementById("modeSwitch");
        // Фіксуємо режим слайдеру до зміни для можливості відкату на випадок помилки.
        const oldChecked = !modeSwitch.checked;
        const newMode = modeSwitch.checked ? "AUTO" : "MANUAL";

        try {
          const res = await fetch(`${ESP_URL}/set_mode`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ mode: newMode })
          });
          if (!res.ok) throw new Error("Server error");

          // Якщо режим змінено успішно

          await synchronize();
        } catch (err) {
          // Помилка підключення (візуально повернути слайдер на місце)
          modeSwitch.checked = oldChecked;
          alert("No connection. Try again");
        }
      }

      function timeToSeconds(timeStr) {
        const [hours, minutes] = timeStr.split(":").map(Number);
        return hours * 3600 + minutes * 60;
      }

       document.getElementById("toggleBtn").addEventListener("click", toggleRelay);
      document.getElementById("setAutoBtn").addEventListener("click", setAutoMode);
      document.getElementById("modeSwitch").addEventListener("change", changeMode);

      // При завантаженні сторінки синхронізуємо плату та сайт
      synchronize();
