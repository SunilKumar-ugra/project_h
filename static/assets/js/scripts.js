$(document).ready(function () {
    $('#open_camera').click((e) =>{
        e.preventDefault()
        $('.capturing__image_block').show()
        $('#camera-and-choose-image-block').hide()
        $('.webcam__block').show()

        const video = document.getElementById('video')
        Promise.all([
            faceapi.nets.tinyFaceDetector.loadFromUri('..\\static\\assets\\js\\models\\'),
            faceapi.nets.faceLandmark68Net.loadFromUri('..\\static\\assets\\js\\models\\'),
            faceapi.nets.faceRecognitionNet.loadFromUri( '..\\static\\assets\\js\\models\\'),
            faceapi.nets.faceExpressionNet.loadFromUri(  '..\\static\\assets\\js\\models\\'),
            faceapi.nets.ageGenderNet.loadFromUri('..\\static\\assets\\js\\models\\')
        ]).then(startVideo)

        function startVideo() {
            navigator.getUserMedia(
                { video: {} },
                stream => video.srcObject = stream,
                err => console.error(err)
            )
        }

        let maximunProb = ""

        video.addEventListener('play', () => {
            const canvas = faceapi.createCanvasFromMedia(video)
            document.body.append(canvas)
            const displaySize = { width: video.width, height: video.height }
            faceapi.matchDimensions(canvas, displaySize)
            setTimeout(async () => {
                const detections = await faceapi.detectAllFaces(video, new faceapi.TinyFaceDetectorOptions()).withFaceLandmarks().withFaceExpressions().withAgeAndGender()
                const resizedDetections = await faceapi.resizeResults(detections, displaySize)
                let detectedResult = detections[0].expressions
                console.log(detectedResult)
                console.log(typeof detectedResult)
                maximunProb = Object.keys(detectedResult).reduce((a, b) => detectedResult[a] > detectedResult[b] ? a : b);
                let acc = 0
                Object.values(detectedResult).forEach(value => {
                    console.log(value)

                    if(acc < value) acc = value
                })
                console.log(maximunProb)
                console.log(acc)

                // $('.capturing__image_block').hide()
                $('.progress').hide()
                $('.capturing__image_block').css('margin-top', '16%');
                $('.capturing__image_block').css('margin-left', '-10%');

                $('#capturing__image_result').html('THE RECOGNIZED IMAGE IS ' + maximunProb.toLocaleUpperCase() + ' WITH ACCURACY OF ' + acc.toFixed(2)*100 + '% AND THE SONG IS PLAYED IN '+ maximunProb.toLocaleUpperCase() + ' MOOD')

                const audio = new Audio('..\\static\\assets\\js\\songs\\' + maximunProb + '/' + maximunProb + '.mp3')
                audio.play()
                canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height)
                faceapi.draw.drawDetections(canvas, resizedDetections)
                // faceapi.draw.drawFaceLandmarks(canvas, resizedDetections)
                faceapi.draw.drawFaceExpressions(canvas, resizedDetections)

            }, 5000)
        })
        $('#view-graphs-2').show()
        $('#stop__music_btn2').show()
    })

    $('#choose_image').click((e) => {
        e.preventDefault()
        $('#upload_image').trigger('click')
    })

    $("input[id='upload_image']").on('change', function (event) {
        $('#camera-and-choose-image-block').hide()
        $('#image-preview-block').show()
        $('#process__image_btn').show()
        let input = this;
        let reader = new FileReader();
        reader.onload = function (e) {
            $('#preview-image').css('marginTop', '-10%')
            $('#preview-image').addClass('img-thumbnail')
            $('#preview-image').attr('src', e.target.result);
        }
        reader.readAsDataURL(input.files[0]);
    })

    $('#process-img').click((e) => {
        e.preventDefault()

        $('#loader').show()

        let image = $('#upload_image').prop('files')[0]
        let data = image['name'];

        $.ajax({
            url: "http://127.0.0.1:8000/process-image/",
            type: "POST",
            dataType: 'json',
            data: {
                image: data,
                csrfmiddlewaretoken: '{{ csrf_token }}'
            },
            success: function (xhr) {
                $('#loader').hide()
                let result = xhr.responseText
                console.log(result)
            },
            error: function (xhr) {
                $('#loader').hide()
                let result = xhr.responseText
                document.body.style.backgroundImage = "url('background-image','../images/home/WhatsApp Image 2020-09-11 at 12.15.08 AM (2).jpeg')"
                $('#choose-image-reslt').html("The recognized emotion is <b>" + result + "</b>. The song is played with " + result + " mood")
                $('#process-img').hide()
                $('#view-graphs-1').show()
                $('#stop__music_btn1').show()

                const audio = new Audio
                ('..\\static\\assets\\js\\songs\\' + result + '/' + result + '.mp3')
                audio.play()

                //let song_path = '{% static "assets/js/songs/"' + result + '/' + result + '.mp3' + '%}'
               // const audio = new Audio(decodeURI(song_path))
                //audio.play()
            }
        })
    })

    $('#upload-new-song').click((e) => {
        e.preventDefault();

        $('#loader').show()

        let song_file = $('.file__upload').prop('files')[0]
        let data = song_file['name'];



        $.ajax({
            url: "http://127.0.0.1:8000/upload-new-song/",
            type: "POST",
            dataType: 'json',
            data: {
                song: data,
                folder: folder_name,
                csrfmiddlewaretoken: '{{ csrf_token }}'
            },
            success: function (xhr) {
                $('#loader').hide()
                let result = xhr.responseText
                console.log(result)
            },
            error: function (xhr) {
                $('#loader').hide()
                let result = xhr.responseText
                console.log(result)
                if(!alert("song uploaded")){window.location.reload()}

                $('#add-new-song-modal').modal('hide')
            }
        })

    //     function validateSong(data) {
    //         let name = data.split('.')[0]
    //
    //         let is_valid = name.charAt(1) == 'H' || name.charAt(1) == 'S' || name.charAt(1) == 'A' || name.charAt(1) == 'D' || name.charAt(1) == 'F' || name.charAt(1) == 'N' || name.charAt(1) == 'S' && name.charAt(2) == 'U'
    //
    //         let folder_name = ""
    //
    //         if(name.charAt(1) == 'H') folder_name = "happy"
    //         if(name.charAt(1) == 'S') folder_name = "sad"
    //         if(name.charAt(1) == 'A') folder_name = "angry"
    //         if(name.charAt(1) == 'D') folder_name = "disgust"
    //         if(name.charAt(1) == 'N') folder_name = "neutral"
    //         if(name.charAt(1) == 'S' && name.charAt(2) == 'U') folder_name = "surprise"
    //
    //         return [is_valid, folder_name]
    //
    //     }
     })

    $('#choose_song_category').click(function(){
        $('#camera-and-choose-image-block').hide();
        $('.select__song_category_block').show()
    })


    $('#stop__music_btn1').click((e) => {
        e.preventDefault()

        window.location.reload()
    })

    $('#stop__music_btn2').click((e) => {
        e.preventDefault()

        window.location.reload()
    })
})