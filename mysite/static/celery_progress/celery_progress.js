var CeleryProgressBar = (function () {
    function onSuccessDefault(progressBarElement, progressBarMessageElement) {
        progressBarElement.style.backgroundColor = '#76ce60';
        progressBarMessageElement.innerHTML = "Success!";
        // const Http = new XMLHttpRequest();
        // const url='http://localhost:8000/visualizedata/visualizePredictionResults';
        // Http.open("GET", url);
        // Http.send();
        // Http.onreadystatechange=(e)=>{
        //     console.log(Http.responseText)
        // }

        // location.href = location.href + "/visualizePredictionResults";
        // location.replace("http://localhost:8000/visualizedata?parameter=0");
        // location.reload();
        document.getElementById("paragraph").innerHTML = "Prueba JS";

        $ajax({
            type: "POST",
            url: '{{ url 'visualizePredictionResults'}}',   
            data: {csrfmiddlewaretoken: '{{ csrf_token }}',
                  text: "this is my test view"},   /* Passing the text data */
            success:  function(response){
                   alert(response);
               }
        });
        // debugger;
    }

    function getContext(params) {
        console.log(params)
    }

    function onErrorDefault(progressBarElement, progressBarMessageElement) {
        progressBarElement.style.backgroundColor = '#dc4f63';
        progressBarMessageElement.innerHTML = "Uh-Oh, something went wrong!";
    }

    function onProgressDefault(progressBarElement, progressBarMessageElement, progress) {
        progressBarElement.style.backgroundColor = '#68a9ef';
        progressBarElement.style.width = progress.percent + "%";
        progressBarMessageElement.innerHTML = progress.current + ' of ' + progress.total + ' processed.';
    }

    function updateProgress (progressUrl, options) {
        options = options || {};
        var progressBarId = options.progressBarId || 'progress-bar';
        var progressBarMessage = options.progressBarMessageId || 'progress-bar-message';
        var progressBarElement = options.progressBarElement || document.getElementById(progressBarId);
        var progressBarMessageElement = options.progressBarMessageElement || document.getElementById(progressBarMessage);
        var onProgress = options.onProgress || onProgressDefault;
        var onSuccess = options.onSuccess || onSuccessDefault;
        var onError = options.onError || onErrorDefault;
        var pollInterval = options.pollInterval || 500;

        fetch(progressUrl).then(function(response) {
            response.json().then(function(data) {
                if (data.progress) {
                    onProgress(progressBarElement, progressBarMessageElement, data.progress);
                }
                if (!data.complete) {
                    setTimeout(updateProgress, pollInterval, progressUrl, options);
                } else {
                    if (data.success) {
                        onSuccess(progressBarElement, progressBarMessageElement);
                    } else {
                        onError(progressBarElement, progressBarMessageElement);
                    }
                }
            });
        });
    }
    return {
        onSuccessDefault: onSuccessDefault,
        onErrorDefault: onErrorDefault,
        onProgressDefault: onProgressDefault,
        updateProgress: updateProgress,
        initProgressBar: updateProgress,  // just for api cleanliness
    };
})();
