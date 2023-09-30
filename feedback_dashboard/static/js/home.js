
$(document).ready(function() {
    $("#generate_button").click(function() {
        // Disable the button
        $(this).prop("disabled", true);

        // Show the loading text
        $(".loading-text").text("Loading...");
        $(".loading-text").show();
        $.ajax({
        type: 'GET',
        url: "/generate/",
        success: function(data) {
                // Handle the response data here
                console.log(data);
                $(".loading-text").hide();
                $(".download-icon").show();
          },
        error: function(jqXHR) {
             console.log("error")
             var errorResponse = JSON.parse(jqXHR.responseText);
             $(".loading-text").text(errorResponse.error);
        }
        })
    });

    $("#download").click(function() {
        var csvFileUrl = '/media/webservice_output/dashboard_input.csv';
         window.location.href = csvFileUrl;
     });

     const fileInput = $("#fileInput");
     const uploadButton = $("#uploadButton");
     fileInput.change(function() {
        if (fileInput[0].files.length > 0) {
            uploadButton.prop("disabled", false);
        } else {
            uploadButton.prop("disabled", true);
        }
    });

    function updateFileFormatInfo() {
                var fileFormatInfo = $(".file-format-info");
                var annotationsRadio = $("#annotations_radio");
                var configurationRadio = $("#configuration_radio");

                if (annotationsRadio.is(":checked")) {
                    fileFormatInfo.text("Supported file formats: CSV (.csv)");
                } else if (configurationRadio.is(":checked")) {
                    fileFormatInfo.text("Supported file formats: Json (.json)");
                }
            }
    $("#annotations_radio, #configuration_radio").change(updateFileFormatInfo);

});