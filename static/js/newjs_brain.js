$(document).ready(function () {
    // Init
    $('.image-section_brain').hide();
    $('.loader').hide();
    $('#result_brain').hide();
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview_brain').attr( 'src', e.target.result );
            }
            reader.readAsDataURL(input.files[0]);
        }
    }
    $("#imageUpload_brain").change(function () {
        $('.image-section').show();
        $('#btn-predict_brain').show();
        $('#result_brain').text('');
        $('#result_brain').hide();
        readURL(this);
    });
    // Predict
    $('#btn-predict_brain').click(function () {
        var form_data_brain = new FormData($('#upload-file_brain')[0]);

        // Show loading animation
        $(this).hide();
        $('.loader').show();

        // Make prediction by calling api /predict
        $.ajax({
            type: 'POST',
            url: '/predict_brain',
            data: form_data_brain,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                // Get and display the result
                $('.loader').hide();
                $('#result_brain').fadeIn(600);
                $('#result_brain').text(' Sonuç:  ' + data);
                console.log('Success!');
            },
        });
    });


});