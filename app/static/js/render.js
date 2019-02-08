function again(){
	var url = document.getElementById('url');

  document.getElementById('post').innerHTML = 'Getting Results..Please Wait';
	document.getElementById('para').innerHTML = 'Getting Results..Please Wait';

  var request = new XMLHttpRequest();
    request.onreadystatechange = function() { 
    if (request.readyState == 4 && request.status == 200)
        var data = JSON.parse(request.responseText);
        var comments = [];
      
        if (data == 'Invalid Input'){
        document.getElementById('post').innerHTML = 'Please give a valid url';
        document.getElementById('author').innerHTML = '';
        document.getElementById('para').innerHTML = '';
        document.getElementById('date').innerHTML = '';
        document.getElementById('num_comm').innerHTML = '';
        document.getElementById('score').innerHTML = '';
        document.getElementById('flair').innerHTML = '';
        document.getElementById('label').innerHTML = '';
        }
        else{
        var title = data['title'];
        var self_text = data['selftext']
        var author = data['author']
        var flair = data['flair']
        var comments = data['comments']
        var date = data['date']
        var num_comments = data['num_comments']
        var score = data['score']
        var img_src = data['image_src']
        var label = data['label']

    	  document.getElementById('post').innerHTML = title;
        document.getElementById('author').innerHTML = 'By: ' + author;
        document.getElementById('para').innerHTML = self_text;
        document.getElementById('date').innerHTML = 'Posted on: ' + date;
        document.getElementById('num_comm').innerHTML = 'Total comments: '+ num_comments;
        document.getElementById('score').innerHTML = 'Score: ' + score;
        document.getElementById('flair').innerHTML = 'True flair: ' + flair;
        document.getElementById('label').innerHTML = 'Predicted flair: ' + label;
        
        document.getElementById('l1').innerHTML = comments[0];
        document.getElementById('l2').innerHTML = comments[1];
        document.getElementById('l3').innerHTML = comments[2];
        document.getElementById('l4').innerHTML = comments[3];
        document.getElementById('l5').innerHTML = comments[4];
        document.getElementById('l6').innerHTML = comments[5];
        document.getElementById('l7').innerHTML = comments[6];
        document.getElementById('l8').innerHTML = comments[7];
        document.getElementById('l9').innerHTML = comments[8];
        document.getElementById('l10').innerHTML = comments[9];

        if (img_src){
        var x = document.getElementById('i');
        x.height = "500";
        x.width = "500";
        x.src = 'https://i.redd.it/' + img_src[0] + '.' + img_src[1];
      }

      }
    };
    const endpoint = '/render'
    request.open("POST", endpoint, true); 
    request.setRequestHeader('Access-Control-Allow-Origin', '*');
    request.setRequestHeader('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept')
    request.setRequestHeader('Content-Type', 'application/json');
    request.send(url.value);
}