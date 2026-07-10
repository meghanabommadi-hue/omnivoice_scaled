"""
Test OmniVoice zero-shot voice cloning across multiple Indian languages,
with pure-native-script and English-mixed (code-switched) sentences in the
EMI / debt-collection reminder call domain. Each sentence addresses a
different (deliberately tricky/uncommon) Indian name.
"""

from omnivoice import OmniVoice
import soundfile as sf
import torch
import os

REF_AUDIO = "audios/reference_audios/saavi_vb.wav"
REF_TEXT = "hello sir, i hope sab theek chal raha hoga, batayiye mein aapki kis tarah se madad kar sakti hun"
OUTPUT_DIR = "audios/output_audios/emi_indian_languages"

# Each entry: (language_code, label, [5 sentences, each addressing a different Indian name])
TEST_CASES = [
    (
        "hi",
        "hindi",
        [
            "नमस्ते वेंकटेशवरन जी, आपकी EMI इस महीने अभी तक जमा नहीं हुई है, कृपया जल्द भुगतान करें।",
            "श्रीमती धनलक्ष्मी सुब्रमण्यम, आपके loan account पर penalty लग सकती है, कृपया due date से पहले payment कर दें।",
            "नमस्कार अरुणाचलम पिल्लई साहब, पिछली बार आपने जो promise किया था वो payment अभी तक नहीं आई है।",
            "Chidambaranathan ji, aapki 3 EMI installments overdue hain, please turant settlement kar dijiye.",
            "Radhakrishnan Iyer sir, agar aaj payment nahi hui to aapka case legal team ko bhेज diya jayega.",
        ],
    ),
    (
        "ta",
        "tamil",
        [
            "வணக்கம் திருவாளர் வெங்கடசுப்ரமணியம், உங்கள் EMI கடந்த மாதம் இருந்து நிலுவையில் உள்ளது.",
            "திருமதி பத்மநாபன் ராஜலட்சுமி, தயவுசெய்து இன்றே உங்கள் loan account balance-ஐ செலுத்துங்கள்.",
            "Meenakshisundaram ஐயா, நீங்க last call-ல சொன்ன promise படி payment இன்னும் வரல, penalty charges apply ஆகலாம்.",
            "வணக்கம் Dhanushkodi Alagarsamy sir, உங்கள் account-ல் overdue amount உள்ளது, உடனே settle செய்யுங்கள்.",
            "Sir Kanagasabapathy, please ஒரு confirmed date sollunga, illana legal notice anuppanum.",
        ],
    ),
    (
        "te",
        "telugu",
        [
            "నమస్కారం శ్రీ వెంకటరమణ చౌదరి గారు, మీ EMI ఈ నెల ఇంకా చెల్లించలేదు, దయచేసి వెంటనే చెల్లించండి.",
            "సుబ్రహ్మణ్యేశ్వరరావు గారు, మీ loan account మీద penalty charges apply అవ్వొచ్చు, దయచేసి due date లోపు కట్టండి.",
            "Padmavathi Venkataramanaiah గారు, మీరు last call లో ఇచ్చిన promise ప్రకారం payment ఇంకా రాలేదు.",
            "Ramalingeswara Rao sir, meeru 2 installments miss chesaru, please final settlement date cheppandi.",
            "నమస్కారం Lakshminarasimha Murthy గారు, ఈ రోజు payment కాకపోతే legal team ki forward chestamu.",
        ],
    ),
    (
        "mr",
        "marathi",
        [
            "नमस्कार श्री वेंकटेशकुमार देशपांडे, तुमची EMI या महिन्यात अजून जमा झालेली नाही, कृपया लवकर भरा.",
            "सौ. पद्मिनीराजे भोसले, तुमच्या loan account वर penalty लागू शकते, कृपया due date आधी payment करा.",
            "Dhananjayrao Kulkarni साहेब, तुम्ही मागच्या call मध्ये दिलेला promise अजून पूर्ण झाला नाही.",
            "Yashwantrao Chavan sir, tumchya account var 2 EMI installments overdue aahet, please lवकर settle करा.",
            "नमस्कार Ramchandra Vishwanath Deshmukh, aaj payment nahi zali tar case legal team kade jail.",
        ],
    ),
    (
        "bn",
        "bengali",
        [
            "নমস্কার শ্রী বেঙ্কটেশ্বর মুখোপাধ্যায়, আপনার EMI এই মাসে এখনো জমা হয়নি, অনুগ্রহ করে দ্রুত পরিশোধ করুন।",
            "শ্রীমতী পদ্মাবতী চট্টোপাধ্যায়, আপনার loan account-এ penalty প্রযোজ্য হতে পারে, please due date-এর আগে payment করুন।",
            "Dhritikanta Bandyopadhyay বাবু, আপনি গত call-এ যে promise দিয়েছিলেন সেই payment এখনো আসেনি।",
            "Ashutosh Chakraborty sir, apnar 2ta EMI installment overdue ache, please turanto settlement korun।",
            "নমস্কার Bibhutibhushan Mukhopadhyay, aaj payment na hole apnar case legal team-e forward kore deba।",
        ],
    ),
    (
        "gu",
        "gujarati",
        [
            "નમસ્તે શ્રી વેંકટપ્રસાદ ઠાકોર, તમારી EMI આ મહિને હજુ જમા થઈ નથી, કૃપા કરીને વહેલી તકે ભરો.",
            "શ્રીમતી પદ્માવતીબેન વ્યાસ, તમારા loan account પર penalty લાગુ થઈ શકે છે, please due date પહેલા payment કરો.",
            "Dhirubhai Jayantilal Zaveri saheb, tame gaya call ma je promise aapyu hatu te payment hજુ aavyu nathi.",
            "Rajnikant Purushottam Trivedi sir, tamara 2 EMI installments overdue che, please jaldi settlement karo.",
            "નમસ્તે Yagnesh Chandulal Oza, aaje payment nahi thay to case legal team ne forward kari devama aavshe.",
        ],
    ),
    (
        "kn",
        "kannada",
        [
            "ನಮಸ್ಕಾರ ಶ್ರೀ ವೆಂಕಟಸುಬ್ರಹ್ಮಣ್ಯ ಹೆಗಡೆ, ನಿಮ್ಮ EMI ಈ ತಿಂಗಳು ಇನ್ನೂ ಪಾವತಿಯಾಗಿಲ್ಲ, ದಯವಿಟ್ಟು ಬೇಗ ಪಾವತಿಸಿ.",
            "ಶ್ರೀಮತಿ ಪದ್ಮಾವತಮ್ಮ ಜೋಯಿಸ್, ನಿಮ್ಮ loan account ಮೇಲೆ penalty ಅನ್ವಯಿಸಬಹುದು, please due date ಮೊದಲು payment ಮಾಡಿ.",
            "Dhruvanarayana Bhattacharya avare, nivu last call nalli kotta promise indaage payment innu bandilla.",
            "Raghavendra Achutarama Rao sir, nimma 2 EMI installments overdue ide, please bega settlement madi.",
            "ನಮಸ್ಕಾರ Kariyappa Nanjundaswamy, indu payment aagadidre case legal team ge forward madtivi.",
        ],
    ),
    (
        "ml",
        "malayalam",
        [
            "നമസ്കാരം ശ്രീ വെങ്കിടസുബ്രഹ്മണ്യൻ നമ്പ്യാർ, നിങ്ങളുടെ EMI ഈ മാസം ഇതുവരെ അടച്ചിട്ടില്ല, ദയവായി വേഗം അടയ്ക്കുക.",
            "ശ്രീമതി പത്മനാഭൻ രാധാമണി, നിങ്ങളുടെ loan account-ൽ penalty ബാധകമായേക്കാം, please due date മുൻപ് payment ചെയ്യുക.",
            "Dhanuvachapuram Krishnankutty Nair sir, ningal kഴിഞ്ഞ call-il paranja promise pole payment ippozhum vannilla.",
            "Achuthanandan Parameswara Pillai avare, ningalude 2 EMI installments overdue aanu, please veഗam settle cheyyu.",
            "നമസ്കാരം Vasudevan Namboothirippad, innu payment illenkil case legal team-inu forward cheyyum.",
        ],
    ),
    (
        "pa",
        "punjabi",
        [
            "ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ ਸਰਦਾਰ ਵੈਂਕਟਪ੍ਰਤਾਪ ਸਿੰਘ ਔਲਖ, ਤੁਹਾਡੀ EMI ਇਸ ਮਹੀਨੇ ਹਾਲੇ ਜਮ੍ਹਾਂ ਨਹੀਂ ਹੋਈ, ਕਿਰਪਾ ਕਰਕੇ ਜਲਦੀ ਭਰੋ।",
            "ਬੀਬੀ ਪਦਮਿੰਦਰ ਕੌਰ ਬਰਾੜ, ਤੁਹਾਡੇ loan account ਤੇ penalty ਲੱਗ ਸਕਦੀ ਹੈ, please due date ਤੋਂ ਪਹਿਲਾਂ payment ਕਰੋ।",
            "Dharampreet Singh Chahal ji, tusi pichli call te jo promise kita si ohde mutabik payment aje vi nahi aayi.",
            "Rajwinder Kaur Aulakh sir, tuhade 2 EMI installments overdue hain, please jaldi settlement karo.",
            "ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ Baljinder Singh Dhillon, awen payment nahi hoyi te case legal team nu forward kar dita javega.",
        ],
    ),
    (
        "ur",
        "urdu",
        [
            "السلام علیکم جناب وینکٹ پرساد قریشی، آپ کی EMI اس مہینے ابھی تک جمع نہیں ہوئی، براہ کرم جلد ادا کریں۔",
            "محترمہ پدماوتی بیگم صدیقی، آپ کے loan account پر penalty لاگو ہو سکتی ہے، please due date سے پہلے payment کریں۔",
            "Zaheeruddin Ahmed Farooqui sahab, aap ne pichli call mein jo promise kiya tha us ke mutabiq payment abhi tak nahi aayi.",
            "Wajahat Hussain Chishti sir, aap ki 2 EMI installments overdue hain, please jaldi settlement kar dein.",
            "السلام علیکم Nasreen Fatima Abidi, aaj payment na hui to case legal team ko forward kar diya jayega.",
        ],
    ),
]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice",
        device_map="cuda:0",
        dtype=torch.float16,
    )

    for lang_code, label, sentences in TEST_CASES:
        texts = sentences
        languages = [lang_code] * len(texts)
        ref_texts = [REF_TEXT] * len(texts)
        ref_audios = [REF_AUDIO] * len(texts)

        audios = model.generate(
            text=texts,
            language=languages,
            ref_text=ref_texts,
            ref_audio=ref_audios,
        )

        for idx, audio in enumerate(audios, start=1):
            out_path = os.path.join(OUTPUT_DIR, f"{label}_{idx}.wav")
            sf.write(out_path, audio, 24000)
            print(f"[{label}/{idx}] wrote {out_path}")


if __name__ == "__main__":
    main()
